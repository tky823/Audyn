import importlib
import os
import warnings
from typing import Dict, Iterable, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer

from ... import __version__ as _version
from ...criterion.gan import GANCriterion
from ...metrics import MeanMetric
from ...models.gan import BaseGAN
from ...optim.lr_scheduler import GANLRScheduler
from ...optim.optimizer import GANOptimizer, MovingAverageWrapper, MultiOptimizers
from ...utils import instantiate
from ..clip_grad import GANGradClipper
from ..data import BaseDataLoaders
from ..hydra.utils import instantiate_grad_clipper
from ..model import unwrap
from .base import BaseGenerator, BaseTrainer, _is_audyn_clip_gradient, _is_torch_clip_gradient


class GANTrainer(BaseTrainer):
    model: BaseGAN
    optimizer: GANOptimizer
    lr_scheduler: GANLRScheduler
    grad_clipper: GANGradClipper
    criterion: GANCriterion

    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: BaseGAN,
        optimizer: GANOptimizer,
        lr_scheduler: Optional[GANLRScheduler] = None,
        grad_clipper: Optional[GANGradClipper] = None,
        criterion: GANCriterion = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            grad_clipper=grad_clipper,
            criterion=criterion,
            config=config,
        )

    def run(self) -> None:
        train_key, validation_key = "train", "validation"
        generator_key, discriminator_key = "generator", "discriminator"

        start_epoch_idx = self.epoch_idx

        for epoch_idx in range(start_epoch_idx, self.epochs):
            train_loss = self.train_one_epoch()

            if isinstance(self.optimizer.generator, MovingAverageWrapper):
                self.logger.info("Set moving average model of generator.")
                self.optimizer.generator.set_moving_average_model()

            if isinstance(self.optimizer.discriminator, MovingAverageWrapper):
                self.logger.info("Set moving average model of discriminator.")
                self.optimizer.discriminator.set_moving_average_model()

            validation_loss = self.validate_one_epoch()
            self.infer_one_batch()

            if isinstance(self.optimizer.generator, MovingAverageWrapper):
                self.logger.info("Remove moving average model of generator.")
                self.optimizer.generator.remove_moving_average_model()

            if isinstance(self.optimizer.discriminator, MovingAverageWrapper):
                self.logger.info("Remove moving average model of discriminator.")
                self.optimizer.discriminator.remove_moving_average_model()

            total_loss = self.display_loss(train_loss, validation_loss)

            for model_type in [generator_key, discriminator_key]:
                criterion_names = self.criterion_names(self.config.criterion[model_type])

                for criterion_name in criterion_names:
                    self.write_scalar_if_necessary(
                        f"{criterion_name} (epoch)/{train_key}",
                        train_loss[model_type][criterion_name],
                        global_step=self.epoch_idx + 1,
                    )
                    self.write_scalar_if_necessary(
                        f"{criterion_name} (epoch)/{validation_key}",
                        validation_loss[model_type][criterion_name],
                        global_step=self.epoch_idx + 1,
                    )

            for key in [train_key, validation_key]:
                for model_type in [generator_key, discriminator_key]:
                    self.write_scalar_if_necessary(
                        f"total {model_type} (epoch)/{key}",
                        total_loss[key][model_type],
                        global_step=self.epoch_idx + 1,
                    )

            self.epoch_idx += 1

            if (
                hasattr(self.config.train.output.save_checkpoint, "epoch")
                and self.config.train.output.save_checkpoint.epoch
            ):
                save_config = self.config.train.output.save_checkpoint.epoch

                if self.epoch_idx % save_config.every == 0:
                    save_path = save_config.path.format(epoch=self.epoch_idx)
                    self.save_checkpoint_if_necessary(save_path)

            if (
                hasattr(self.config.train.output.save_checkpoint, "last")
                and self.config.train.output.save_checkpoint.last
            ):
                save_config = self.config.train.output.save_checkpoint.last
                save_path = save_config.path.format(
                    epoch=self.epoch_idx, iteration=self.iteration_idx
                )
                self.save_checkpoint_if_necessary(save_path)

    def train_one_epoch(self) -> Dict[str, float]:
        """Train model for one epoch."""
        criterion_config = self.config.criterion
        generator_key_mapping = self.config.train.key_mapping.train.generator
        discriminator_key_mapping = self.config.train.key_mapping.train.discriminator
        lr_scheduler_step_config = self.config.train.steps.lr_scheduler
        train_key = "train"
        generator_key, discriminator_key = "generator", "discriminator"

        generator_criterion_names = self.criterion_names(criterion_config.generator)
        discriminator_criterion_names = self.criterion_names(criterion_config.discriminator)
        generator_mean_metrics = {
            criterion_name: MeanMetric(device=self.device)
            for criterion_name in generator_criterion_names
        }
        discriminator_mean_metrics = {
            criterion_name: MeanMetric(device=self.device)
            for criterion_name in discriminator_criterion_names
        }

        n_batch = 0
        n_remain = self.iteration_idx % len(self.loaders.train)

        self.set_epoch_if_necessary(self.epoch_idx)
        self.unwrapped_model.generator.train()
        self.unwrapped_model.discriminator.train()

        for named_batch in self.loaders.train:
            if n_remain > 0:
                # When checkpoint is a specific iteration,
                # we have to skip the batches we've already treated.
                n_remain -= 1
                continue

            named_batch = self.move_data_to_device(named_batch, self.device)
            named_noise = self.map_to_named_input(
                named_batch,
                key_mapping=generator_key_mapping,
            )
            named_generator_target = self.map_to_named_target(
                named_batch,
                config=criterion_config.generator,
            )
            named_discriminator_target = self.map_to_named_target(
                named_batch,
                config=criterion_config.discriminator,
            )
            fake = self.unwrapped_model.generator(**named_noise)
            named_fake = self.map_to_named_output(
                fake,
                key_mapping=generator_key_mapping,
            )
            named_fake_input = self.map_to_named_input(
                named_fake,
                key_mapping=discriminator_key_mapping.fake,
            )
            named_real_input = self.map_to_named_input(
                named_batch,
                key_mapping=discriminator_key_mapping.real,
            )

            # detach graph for computational efficiency
            named_fake_input_no_grad = {}

            for key in named_fake_input.keys():
                named_fake_input_no_grad[key] = named_fake_input[key].detach()

            fake_output = self.unwrapped_model.discriminator(**named_fake_input_no_grad)
            real_output = self.unwrapped_model.discriminator(**named_real_input)

            named_fake_output = self.map_to_named_output(
                fake_output,
                key_mapping=discriminator_key_mapping.fake,
            )
            named_real_output = self.map_to_named_output(
                real_output,
                key_mapping=discriminator_key_mapping.real,
            )

            assert (
                set(named_fake_output.keys()) & set(named_real_output.keys()) == set()
            ), "named_fake_output and named_real_output should be disjointed."

            named_output = {}
            named_output.update(named_fake_output)
            named_output.update(named_real_output)
            named_estimated = self.map_to_named_estimated(
                named_output,
                config=criterion_config.discriminator,
            )

            total_discriminator_loss = 0
            discriminator_loss = {}

            for criterion_name in discriminator_criterion_names:
                weight = criterion_config.discriminator[criterion_name].weight
                discriminator_loss[criterion_name] = self.criterion.discriminator[criterion_name](
                    **named_estimated[criterion_name],
                    **named_discriminator_target[criterion_name],
                )
                total_discriminator_loss = (
                    total_discriminator_loss + weight * discriminator_loss[criterion_name]
                )
                discriminator_mean_metrics[criterion_name].update(
                    discriminator_loss[criterion_name].item()
                )

                self.write_scalar_if_necessary(
                    f"{criterion_name} (iteration)/{train_key}",
                    discriminator_loss[criterion_name].item(),
                    global_step=self.iteration_idx + 1,
                )

            self.write_scalar_if_necessary(
                f"total {discriminator_key} (iteration)/{train_key}",
                total_discriminator_loss,
                global_step=self.iteration_idx + 1,
            )

            self.optimizer.discriminator.zero_grad()
            self.scaler.scale(total_discriminator_loss).backward()
            self.unscale_optimizer_if_necessary(self.optimizer.discriminator)

            if self.grad_clipper is None:
                self.clip_gradient_if_necessary(self.unwrapped_model.discriminator.parameters())
            else:
                self.clip_gradient_if_necessary("discriminator")

            self.optimizer_step(self.optimizer.discriminator)

            if lr_scheduler_step_config and lr_scheduler_step_config.discriminator == "iteration":
                self.lr_scheduler_step(self.lr_scheduler.discriminator, loss=discriminator_loss)

            prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}"
            prompt += f", Iter {self.iteration_idx+1}/{self.iterations}]"
            s = ""

            for criterion_name in discriminator_criterion_names:
                s += f"{criterion_name}: {discriminator_loss[criterion_name]}, "

            s = f"{prompt} {total_discriminator_loss.item()}, {s[:-2]}"

            self.logger.info(s)

            # for updates of generator
            fake_output = self.unwrapped_model.discriminator(**named_fake_input)
            real_output = self.unwrapped_model.discriminator(**named_real_input)

            named_fake_output = self.map_to_named_output(
                fake_output,
                key_mapping=discriminator_key_mapping.fake,
            )
            named_real_output = self.map_to_named_output(
                real_output,
                key_mapping=discriminator_key_mapping.real,
            )

            assert (
                set(named_fake_output.keys()) & set(named_real_output.keys()) == set()
            ), "named_fake_output and named_real_output should be disjointed."
            assert (
                set(named_real_output.keys()) & set(named_fake.keys()) == set()
            ), "named_real_output and named_fake should be disjointed."
            assert (
                set(named_fake.keys()) & set(named_fake_output.keys()) == set()
            ), "named_fake and named_fake_output should be disjointed."

            named_output = {}
            named_output.update(named_fake_output)
            named_output.update(named_real_output)
            named_output.update(named_fake)

            named_estimated = self.map_to_named_estimated(
                named_output,
                config=criterion_config.generator,
            )

            total_generator_loss = 0
            generator_loss = {}

            for criterion_name in generator_criterion_names:
                weight = criterion_config.generator[criterion_name].weight
                generator_loss[criterion_name] = self.criterion.generator[criterion_name](
                    **named_estimated[criterion_name],
                    **named_generator_target[criterion_name],
                )
                total_generator_loss = (
                    total_generator_loss + weight * generator_loss[criterion_name]
                )
                generator_mean_metrics[criterion_name].update(
                    generator_loss[criterion_name].item()
                )

                self.write_scalar_if_necessary(
                    f"{criterion_name} (iteration)/{train_key}",
                    generator_loss[criterion_name].item(),
                    global_step=self.iteration_idx + 1,
                )

            self.write_scalar_if_necessary(
                f"total {generator_key} (iteration)/{train_key}",
                total_generator_loss,
                global_step=self.iteration_idx + 1,
            )

            self.write_train_duration_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
            )
            self.write_train_spectrogram_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
            )
            self.write_train_waveform_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
            )
            self.write_train_audio_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
            )
            self.write_train_image_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
            )

            self.optimizer.generator.zero_grad()
            self.scaler.scale(total_generator_loss).backward()
            self.unscale_optimizer_if_necessary(self.optimizer.generator)

            if self.grad_clipper is None:
                self.clip_gradient_if_necessary(self.unwrapped_model.generator.parameters())
            else:
                self.clip_gradient_if_necessary("generator")

            self.optimizer_step(self.optimizer.generator)
            self.scaler.update()

            if lr_scheduler_step_config and lr_scheduler_step_config.generator == "iteration":
                self.lr_scheduler_step(self.lr_scheduler.generator, loss=generator_loss)

            prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}"
            prompt += f", Iter {self.iteration_idx+1}/{self.iterations}]"
            s = ""

            for criterion_name in generator_criterion_names:
                s += f"{criterion_name}: {generator_loss[criterion_name]}, "

            s = f"{prompt} {total_generator_loss.item()}, {s[:-2]}"

            self.logger.info(s)

            self.iteration_idx += 1
            n_batch += 1

            if (
                hasattr(self.config.train.output.save_checkpoint, "iteration")
                and self.config.train.output.save_checkpoint.iteration
            ):
                save_config = self.config.train.output.save_checkpoint.iteration

                if self.iteration_idx % save_config.every == 0:
                    save_path = save_config.path.format(iteration=self.iteration_idx)
                    self.save_checkpoint_if_necessary(save_path)

            if self.iteration_idx >= self.iterations:
                # Finish training
                break

        train_loss = {
            generator_key: {},
            discriminator_key: {},
        }

        for criterion_name in generator_criterion_names:
            loss = generator_mean_metrics[criterion_name].compute()
            train_loss[generator_key][criterion_name] = loss.item()

        for criterion_name in discriminator_criterion_names:
            loss = discriminator_mean_metrics[criterion_name].compute()
            train_loss[discriminator_key][criterion_name] = loss.item()

        if lr_scheduler_step_config and lr_scheduler_step_config.generator == "epoch":
            self.lr_scheduler_step(self.lr_scheduler.generator, loss=train_loss[generator_key])

        if lr_scheduler_step_config and lr_scheduler_step_config.discriminator == "epoch":
            self.lr_scheduler_step(
                self.lr_scheduler.discriminator, loss=train_loss[discriminator_key]
            )

        return train_loss

    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str, float]:
        """Validate model for one epoch."""
        criterion_config = self.config.criterion
        generator_key_mapping = self.config.train.key_mapping.validation.generator
        discriminator_key_mapping = self.config.train.key_mapping.validation.discriminator
        generator_key, discriminator_key = "generator", "discriminator"

        generator_criterion_names = self.criterion_names(criterion_config.generator)
        discriminator_criterion_names = self.criterion_names(criterion_config.discriminator)
        generator_mean_metrics = {
            criterion_name: MeanMetric(device=self.device)
            for criterion_name in generator_criterion_names
        }
        discriminator_mean_metrics = {
            criterion_name: MeanMetric(device=self.device)
            for criterion_name in discriminator_criterion_names
        }

        n_batch = 0

        self.unwrapped_model.generator.eval()
        self.unwrapped_model.discriminator.eval()

        for named_batch in self.loaders.validation:
            named_batch = self.move_data_to_device(named_batch, self.device)

            # preparation for forward pass of generator
            named_noise = self.map_to_named_input(
                named_batch,
                key_mapping=generator_key_mapping,
            )
            named_generator_target = self.map_to_named_target(
                named_batch,
                config=criterion_config.generator,
            )
            fake = self.unwrapped_model.generator(**named_noise)
            named_fake = self.map_to_named_output(
                fake,
                key_mapping=generator_key_mapping,
            )

            # preparation for forward pass of discriminator
            named_fake_input = self.map_to_named_input(
                named_fake,
                key_mapping=discriminator_key_mapping.fake,
            )
            named_real_input = self.map_to_named_input(
                named_batch,
                key_mapping=discriminator_key_mapping.real,
            )
            named_discriminator_target = self.map_to_named_target(
                named_batch,
                config=criterion_config.discriminator,
            )

            fake_output = self.unwrapped_model.discriminator(**named_fake_input)
            real_output = self.unwrapped_model.discriminator(**named_real_input)

            named_fake_output = self.map_to_named_output(
                fake_output,
                key_mapping=discriminator_key_mapping.fake,
            )
            named_real_output = self.map_to_named_output(
                real_output,
                key_mapping=discriminator_key_mapping.real,
            )

            assert (
                set(named_fake_output.keys()) & set(named_real_output.keys()) == set()
            ), "named_fake_output and named_real_output should be disjointed."
            assert (
                set(named_real_output.keys()) & set(named_fake.keys()) == set()
            ), "named_real_output and named_fake should be disjointed."
            assert (
                set(named_fake.keys()) & set(named_fake_output.keys()) == set()
            ), "named_fake and named_fake_output should be disjointed."

            named_output = {}
            named_output.update(named_fake_output)
            named_output.update(named_real_output)
            named_output.update(named_fake)
            named_discriminator_estimated = self.map_to_named_estimated(
                named_output,
                config=criterion_config.discriminator,
            )
            named_generator_estimated = self.map_to_named_estimated(
                named_output,
                config=criterion_config.generator,
            )

            discriminator_loss = {}

            for criterion_name in discriminator_criterion_names:
                discriminator_loss[criterion_name] = self.criterion.discriminator[criterion_name](
                    **named_discriminator_estimated[criterion_name],
                    **named_discriminator_target[criterion_name],
                )
                discriminator_mean_metrics[criterion_name].update(
                    discriminator_loss[criterion_name].item()
                )

            generator_loss = {}

            for criterion_name in generator_criterion_names:
                generator_loss[criterion_name] = self.criterion.generator[criterion_name](
                    **named_generator_estimated[criterion_name],
                    **named_generator_target[criterion_name],
                )
                generator_mean_metrics[criterion_name].update(
                    generator_loss[criterion_name].item()
                )

            self.write_validation_duration_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_spectrogram_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_waveform_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_audio_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_image_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )

            n_batch += 1

        validation_loss = {
            generator_key: {},
            discriminator_key: {},
        }

        for criterion_name in generator_criterion_names:
            loss = generator_mean_metrics[criterion_name].compute()
            validation_loss[generator_key][criterion_name] = loss.item()

        for criterion_name in discriminator_criterion_names:
            loss = discriminator_mean_metrics[criterion_name].compute()
            validation_loss[discriminator_key][criterion_name] = loss.item()

        return validation_loss

    @torch.no_grad()
    def infer_one_batch(self) -> None:
        """Inference using one batch."""
        if hasattr(self.config.train.key_mapping, "inference"):
            inference_key_mapping = self.config.train.key_mapping.inference.generator
        elif hasattr(self.config.train.key_mapping, "validation"):
            inference_key_mapping = self.config.train.key_mapping.validation.generator
        else:
            inference_key_mapping = self.config.train.key_mapping.generator

        n_batch = 0

        self.unwrapped_model.generator.eval()

        for named_batch in self.loaders.validation:
            named_batch = self.move_data_to_device(named_batch, self.device)
            named_input = self.map_to_named_input(
                named_batch,
                key_mapping=inference_key_mapping,
            )

            unwrapped_generator = unwrap(self.unwrapped_model.generator)

            if hasattr(unwrapped_generator, "inference"):
                output = unwrapped_generator.inference(**named_input)
            else:
                output = unwrapped_generator(**named_input)

            named_output = self.map_to_named_output(
                output,
                key_mapping=inference_key_mapping,
            )

            self.write_inference_duration_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_spectrogram_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_waveform_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_audio_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_image_if_necessary(
                named_output,
                named_batch,
                config=self.config.train.record,
                batch_idx=n_batch,
            )

            n_batch += 1

            # Process only first batch.
            break

    def count_num_parameters(self, model: nn.Module) -> int:
        """Count number of parameters.

        Args:
            model (nn.Module): Generator or discriminator.

        Returns:
            int: Number of parameters in given model.
        """
        num_parameters = 0

        for p in model.parameters():
            if p.requires_grad:
                num_parameters += p.numel()

        return num_parameters

    def display_model(self, display_num_parameters: bool = True) -> None:
        generator, discriminator = (
            self.unwrapped_model.generator,
            self.unwrapped_model.discriminator,
        )
        self.logger.info(generator)

        if display_num_parameters:
            self.logger.info(
                f"# of generator parameters: {self.count_num_parameters(generator)}.",
            )

        self.logger.info(discriminator)

        if display_num_parameters:
            self.logger.info(
                f"# of discriminator parameters: {self.count_num_parameters(discriminator)}."
            )

    def display_loss(
        self,
        train_loss: Dict[str, Dict[str, float]],
        validation_loss: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Dict[str, float]:
        criterion_config = self.config.criterion
        train_key, validation_key = "train", "validation"
        generator_key, discriminator_key = "generator", "discriminator"
        total_loss = {train_key: 0, validation_key: 0}
        total_loss = {}

        for train_or_validation in [train_key, validation_key]:
            total_loss[train_or_validation] = {}
            for model_type in [generator_key, discriminator_key]:
                total_loss[train_or_validation][model_type] = 0

        prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}] ({train_key})"
        s = ""

        for model_type in [generator_key, discriminator_key]:
            criterion_names = self.criterion_names(criterion_config[model_type])

            for criterion_name in criterion_names:
                weight = criterion_config[model_type][criterion_name].weight
                loss = train_loss[model_type][criterion_name]
                total_loss[train_key][model_type] = (
                    total_loss[train_key][model_type] + weight * loss
                )
                s += f"{criterion_name}: {loss}, "

        s = f"{prompt} {total_loss[train_key][model_type]}, {s[:-2]}"
        self.logger.info(s)

        prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}] ({validation_key})"
        s = ""

        for model_type in [generator_key, discriminator_key]:
            criterion_names = self.criterion_names(criterion_config[model_type])

            for criterion_name in criterion_names:
                weight = criterion_config[model_type][criterion_name].weight
                loss = validation_loss[model_type][criterion_name]
                total_loss[validation_key][model_type] = (
                    total_loss[validation_key][model_type] + weight * loss
                )
                s += f"{criterion_name}: {loss}, "

        s = f"{prompt} {total_loss[validation_key][model_type]}, {s[:-2]}"
        self.logger.info(s)

        return total_loss

    def unscale_optimizer_if_necessary(
        self, optimizer: Optional[Optimizer] = None, enable: bool = True
    ) -> None:
        """Unscale optimizer containing values if necessary.

        Args:
            optimizer (Optimizer, optional): Optimizer to be unscaled.
            enable (bool): If ``True``, ``self.scaler.unscale_`` is
                applied to ``optimizer`` before clipping gradient. This operation
                doesn't anything when ``self.scaler.is_enabled()`` is ``False``.

        """
        if enable:
            if self.scaler.is_enabled() and optimizer is None:
                raise ValueError("optimizer is not given.")

            if isinstance(optimizer, MultiOptimizers):
                for _optimizer in optimizer.optimizers.values():
                    self.scaler.unscale_(_optimizer)
            else:
                self.scaler.unscale_(optimizer)

    def clip_gradient_if_necessary(
        self,
        parameters_or_name: Union[Iterable[torch.Tensor], torch.Tensor, str],
    ) -> None:
        """Clip gradient if self.grad_clipper is given.

        Args:
            parameters_or_name (Iterable of torch.Tensor, torch.Tensor, or str):
                Model parameters (legacy gradient clipping) or
                name of module (generator or discriminator).

        """
        if not hasattr(self.config.train, "clip_gradient"):
            # clip_gradient is not defined.
            return

        clip_gradient_config = self.config.train.clip_gradient

        if not hasattr(clip_gradient_config, "_target_"):
            # clip_gradient is not used.
            return

        clip_gradient_target = clip_gradient_config._target_

        if clip_gradient_target is None or clip_gradient_target == "":
            return

        if self.grad_clipper is not None:
            is_legacy = False
        else:
            mod_name, var_name = clip_gradient_target.rsplit(".", maxsplit=1)
            clip_gradient_fn = getattr(importlib.import_module(mod_name), var_name)

            if _is_audyn_clip_gradient(clip_gradient_fn):
                # for backward compatibility
                # clip all parameters
                self.grad_clipper = instantiate(clip_gradient_config, self.model.parameters())
                is_legacy = False
            elif clip_gradient_fn is GANGradClipper:
                # for backward compatibility
                generator_grad_clipper = instantiate_grad_clipper(
                    clip_gradient_config.generator, self.unwrapped_model.generator.parameters()
                )
                discriminator_grad_clipper = instantiate_grad_clipper(
                    clip_gradient_config.discriminator,
                    self.unwrapped_model.discriminator.parameters(),
                )
                self.grad_clipper = GANGradClipper(
                    generator_grad_clipper, discriminator_grad_clipper
                )
                is_legacy = False
            elif _is_torch_clip_gradient(clip_gradient_fn):
                is_legacy = True
            else:
                raise ValueError("Invalid condition is detected.")

        if is_legacy:
            # for backward compatibility
            parameters = parameters_or_name
            instantiate(clip_gradient_config, parameters)
        else:
            if isinstance(parameters_or_name, str):
                name = parameters_or_name

                if name == "generator":
                    self.grad_clipper.generator.step()
                elif name == "discriminator":
                    self.grad_clipper.discriminator.step()
                else:
                    raise ValueError(f"Invalid name {name} is given.")
            else:
                # If multiple models are used (e.g. generator and discriminator of GAN),
                # the following operation may be redundant.
                warnings.warn(
                    "If grad_clipper changes the internal state (e.g. training step, etc.), "
                    "excessive calls to .step() can lead to inappropriate results."
                )
                self.grad_clipper.step()

    def load_checkpoint(self, path: str) -> None:
        generator_key, discriminator_key = "generator", "discriminator"

        state_dict = torch.load(path, map_location=self.device)

        # model
        unwrapped_generator = unwrap(self.unwrapped_model.generator)
        unwrapped_generator.load_state_dict(state_dict["model"][generator_key])
        unwrapped_discriminator = unwrap(self.unwrapped_model.discriminator)
        unwrapped_discriminator.load_state_dict(state_dict["model"][discriminator_key])

        # optimizer
        self.optimizer.generator.load_state_dict(
            state_dict["optimizer"][generator_key],
        )
        self.optimizer.discriminator.load_state_dict(
            state_dict["optimizer"][discriminator_key],
        )

        # learning rate scheduler
        if self.lr_scheduler.generator is None:
            assert state_dict["lr_scheduler"][generator_key] is None
        else:
            self.lr_scheduler.generator.load_state_dict(
                state_dict["lr_scheduler"][generator_key],
            )

        if self.lr_scheduler.discriminator is None:
            assert state_dict["lr_scheduler"][discriminator_key] is None
        else:
            self.lr_scheduler.discriminator.load_state_dict(
                state_dict["lr_scheduler"][discriminator_key]
            )

        # gradient scaler
        self.scaler.load_state_dict(state_dict["scaler"])

        self.iteration_idx = state_dict["iteration_idx"]
        self.best_loss = state_dict["best_loss"]
        self.epoch_idx = self.iteration_idx // len(self.loaders.train)

    def save_checkpoint(self, save_path: str) -> None:
        generator_key, discriminator_key = "generator", "discriminator"

        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        state_dict = {}

        # model
        unwrapped_generator = unwrap(self.unwrapped_model.generator)
        unwrapped_discriminator = unwrap(self.unwrapped_model.discriminator)
        state_dict["model"] = {
            generator_key: unwrapped_generator.state_dict(),
            discriminator_key: unwrapped_discriminator.state_dict(),
        }

        # optimizer
        state_dict["optimizer"] = {
            generator_key: self.optimizer.generator.state_dict(),
            discriminator_key: self.optimizer.discriminator.state_dict(),
        }

        # learning rate scheduler
        lr_scheduler = self.lr_scheduler
        state_dict_key = "lr_scheduler"
        state_dict[state_dict_key] = {}

        if self.lr_scheduler.generator is None:
            state_dict[state_dict_key][generator_key] = None
        else:
            state_dict[state_dict_key][generator_key] = lr_scheduler.generator.state_dict()

        if lr_scheduler.discriminator is None:
            state_dict[state_dict_key][discriminator_key] = None
        else:
            state_dict[state_dict_key][discriminator_key] = lr_scheduler.discriminator.state_dict()

        # gradient scaler
        state_dict["scaler"] = self.scaler.state_dict()

        state_dict["iteration_idx"] = self.iteration_idx
        state_dict["best_loss"] = self.best_loss
        state_dict["resolved_config"] = OmegaConf.to_container(self.config, resolve=True)

        # moving average model
        if isinstance(self.optimizer.generator, MovingAverageWrapper) or isinstance(
            self.optimizer.discriminator, MovingAverageWrapper
        ):
            # Save state dict of moving averaged model
            state_dict_key = "moving_average_model"
            state_dict[state_dict_key] = {}

            if isinstance(self.optimizer.generator, MovingAverageWrapper):
                self.optimizer.generator.set_moving_average_model()
                unwrapped_generator = unwrap(self.unwrapped_model.generator)
                state_dict[state_dict_key][generator_key] = unwrapped_generator.state_dict()
                self.optimizer.generator.remove_moving_average_model()
            else:
                state_dict[state_dict_key][generator_key] = None

            if isinstance(self.optimizer.discriminator, MovingAverageWrapper):
                self.optimizer.discriminator.set_moving_average_model()
                unwrapped_discriminator = unwrap(self.unwrapped_model.discriminator)
                state_dict[state_dict_key][
                    discriminator_key
                ] = unwrapped_discriminator.state_dict()
                self.optimizer.discriminator.remove_moving_average_model()
            else:
                state_dict[state_dict_key][discriminator_key] = None

        # Store metadata
        module_name, class_name = self.__module__, self.__class__.__name__
        class_name = class_name if module_name is None else f"{module_name}.{class_name}"
        state_dict["_metadata"] = {
            "version": _version,
            "driver": class_name,
        }

        torch.save(state_dict, save_path)

        s = f"Save model: {save_path}."
        self.logger.info(s)


class GANGenerator(BaseGenerator):
    """Waveform synthesis using generator of GAN."""

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        unwrapped_generator = unwrap(self.unwrapped_model.generator)
        unwrapped_discriminator = unwrap(self.unwrapped_model.discriminator)

        unwrapped_generator.load_state_dict(state_dict["model"]["generator"])
        unwrapped_discriminator.load_state_dict(state_dict["model"]["discriminator"])

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        unwrapped_generator = unwrap(self.unwrapped_model.generator)

        for named_data in self.loader:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(
                named_data, key_mapping=self.config.test.key_mapping.inference
            )
            named_identifier = self.map_to_named_identifier(
                named_data, key_mapping=self.config.test.key_mapping.inference
            )

            if hasattr(unwrapped_generator, "inference"):
                output = unwrapped_generator.inference(**named_input)
            else:
                output = unwrapped_generator(**named_input)

            named_output = self.map_to_named_output(
                output, key_mapping=self.config.test.key_mapping.inference
            )

            self.save_inference_audio_if_necessary(
                named_output, named_data, named_identifier, config=self.config.test.output
            )
