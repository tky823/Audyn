import importlib
import os
import subprocess
import warnings
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from omegaconf import DictConfig, OmegaConf
from packaging import version
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from ... import __version__ as _version
from ...metrics import MeanMetric
from ...optim.optimizer import (
    ExponentialMovingAverageCodebookOptimizer,
    MovingAverageWrapper,
    MultiOptimizers,
)
from ...utils.hydra.utils import (
    instantiate,
    instantiate_criterion,
    instantiate_grad_clipper,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
)
from ...utils.model import set_device
from ..alignment import expand_by_duration
from ..clip_grad import GradClipper
from ..data import BaseDataLoaders, select_device
from ..distributed import select_global_rank, select_local_rank
from ..hydra.utils import TORCH_CLIP_GRAD_FN
from ..logging import get_logger
from ..model import unwrap
from ..tensorboard import get_summary_writer
from ._decorator import run_only_master_rank


class BaseDriver:
    model: nn.Module
    optimizer: Optimizer
    lr_scheduler: Optional[_LRScheduler]
    criterion: Dict[str, nn.Module]

    config: DictConfig

    logger: Logger

    def __init__(self) -> None:
        pass

    @property
    def unwrapped_model(self) -> nn.Module:
        """Unwrapped model to access attributes directly."""
        return unwrap(self.model)

    def criterion_names(self, config: Optional[DictConfig] = None) -> List[str]:
        if config is None:
            config = self.config.criterion

        names = [key for key in config.keys() if not key.startswith("_") and not key.endswith("_")]
        names = sorted(names)

        return names

    def count_num_parameters(self) -> int:
        """Count number of parameters.

        Returns:
            int: Number of parameters.
        """
        num_parameters = 0

        for p in self.model.parameters():
            if p.requires_grad:
                num_parameters += p.numel()

        return num_parameters

    def display_model(self, display_num_parameters: bool = True) -> None:
        self.logger.info(self.unwrapped_model)

        if display_num_parameters:
            self.logger.info(f"# of parameters: {self.count_num_parameters()}.")

    def set_system(self, config: Optional[DictConfig] = None) -> None:
        """Set attributes related to config of system.

        Args:
            config (DictConfig, optional): Config of system.

        In this method, the following attributes are set.

        - is_distributed (bool): If True, distributed data parallel is used.
        - local_rank (int, optional): Local rank of distributed data parallel.
            When distributed data parallel is disabled, local_rank is set as None.
        - global_rank (int, optional): Global rank of distributed data parallel.
            When distributed data parallel is disabled, global_rank is set as None.
        - device (str): Device to allocate tensors.
        - enable_amp (bool): If True, automatic mixed precision package (amp) is used.
            This is available when CUDA is also available.

        """
        if config is None:
            config = self.config.system

        self.is_distributed = config.distributed.enable
        self.local_rank = select_local_rank(
            config.accelerator, is_distributed=config.distributed.enable
        )
        self.global_rank = select_global_rank(
            config.accelerator, is_distributed=config.distributed.enable
        )
        self.device = select_device(config.accelerator, is_distributed=config.distributed.enable)

        if hasattr(config, "amp"):
            availability = config.amp.enable

            if availability and not torch.cuda.is_available():
                raise ValueError(
                    "You specied config.system.amp.enable=True, but CUDA is not available."
                )

            if availability is None:
                availability = False

            self.enable_amp = availability
        else:
            self.enable_amp = False

    def move_data_to_device(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        for key in data.keys():
            value = data[key]
            if value is None:
                # None cannot be allocated to specific device.
                pass
            elif isinstance(value, torch.Tensor):
                value = value.to(device)
            elif (
                isinstance(value, int)
                or isinstance(value, float)
                or isinstance(value, str)
                or isinstance(value, list)
                or isinstance(value, dict)
            ):
                # Primitive types cannot be allocated to specific device.
                pass
            else:
                raise TypeError(f"{type(value)} cannot be allocated to device {device}.")

            data[key] = value

        return data

    def map_to_named_input(
        self,
        named_data: Dict[str, torch.Tensor],
        key_mapping: Optional[DictConfig] = None,
        strict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if key_mapping is None:
            key_mapping = self.config.train.key_mapping

        named_input = {}

        for model_key in key_mapping.input.keys():
            data_key = key_mapping.input[model_key]

            _named_data = named_data.get(data_key)

            if _named_data is None:
                if strict:
                    raise ValueError(f"{data_key} is not found in named_data.")
            else:
                named_input[model_key] = _named_data

        return named_input

    def map_to_named_target(
        self,
        named_data: Dict[str, torch.Tensor],
        config: Optional[DictConfig] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Map named batch to named target.

        Args:
            named_data (dict): Dict-type batch.
            config (DictConfig, optional): Config to map batch to target for each criterion.
                Default: ``self.config.criterion``.

        Returns:
            dict: Dict-type named batch. Each key corresponds to name of criterion.

        """
        if config is None:
            config = self.config.criterion

        criterion_names = self.criterion_names(config)
        named_target = {criterion_name: {} for criterion_name in criterion_names}

        for criterion_name in criterion_names:
            key_mapping = config[criterion_name].key_mapping.target

            if key_mapping is None:
                continue

            for criterion_key, data_key in key_mapping.items():
                named_target[criterion_name][criterion_key] = named_data[data_key]

        return named_target

    def map_to_named_output(
        self,
        output: Union[torch.Tensor, Tuple[str, torch.Tensor]],
        key_mapping: Optional[DictConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        """Parse output of model.

        Args:
            output (any): Output of model.
            key_mapping (DictConfig): Config to parse outputs.

        Returns:
            dict: Dict-type data containing output of model.

        """

        def _map(key_or_mapping: Union[str, List[str]], output: Any) -> Dict[str, Any]:
            named_output = {}

            if OmegaConf.is_config(key_or_mapping):
                key_type = type(OmegaConf.to_object(key_or_mapping))
            else:
                key_type = type(key_or_mapping)

            if key_type is str:
                key = key_or_mapping
                named_output[key] = output
            elif key_type is list:
                mapping = key_or_mapping

                for idx in range(len(mapping)):
                    _named_output = _map(mapping[idx], output[idx])
                    assert set(named_output.keys()) & set(_named_output.keys()) == set()
                    named_output.update(_named_output)
            else:
                raise ValueError(f"Invalid key type {key_type} is found.")

            return named_output

        if key_mapping is None:
            key_mapping = self.config.train.key_mapping

        named_output = _map(key_mapping.output, output)

        return named_output

    def map_to_named_estimated(
        self,
        named_output: Dict[str, torch.Tensor],
        config: Optional[DictConfig] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Map named output to dict-type estimated data.

        Args:
            named_output (dict): Dict-type output of model.
            config (DictConfig, optional): Config to map data.

        Returns:
            dict: Dict-type data treated as output of model.
                Each key corresponds to name of criterion.

        """
        if config is None:
            config = self.config.criterion

        criterion_names = self.criterion_names(config)
        named_estimated = {criterion_name: {} for criterion_name in criterion_names}

        for criterion_name in criterion_names:
            key_mapping = config[criterion_name].key_mapping.estimated

            for criterion_key, data_key in key_mapping.items():
                named_estimated[criterion_name][criterion_key] = named_output[data_key]

        return named_estimated

    def map_to_named_identifier(
        self,
        named_data: Dict[str, torch.Tensor],
        key_mapping: Optional[DictConfig] = None,
    ) -> Dict[str, torch.Tensor]:
        """Map dict-type data to filename identifiers.

        Args:
            named_data (dict): Dict-type data given by data loader.

        Returns:
            dict: Filename identifier that maps model outputs to filename to identify data.

        """
        if key_mapping is None:
            key_mapping = self.config.train.key_mapping

        identifier = {}

        for identifier_key in key_mapping.identifier.keys():
            data_key = key_mapping.identifier[identifier_key]
            identifier[identifier_key] = named_data[data_key]

        return identifier


class BaseTrainer(BaseDriver):
    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        grad_clipper: Optional[GradClipper] = None,
        criterion: Dict[str, nn.Module] = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__()

        if isinstance(lr_scheduler, DictConfig):
            lr_scheduler = None

            if config.train.steps.lr_scheduler is not None:
                raise ValueError(
                    "Although learning rate scheduler is not found, its step is specified."
                )

        self.loaders = loaders
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.grad_clipper = grad_clipper
        self.criterion = criterion

        self.config = config

        self._reset(config)

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "BaseTrainer":
        """Instantiate Trainer from config.

        Args:
            config (DictConfig): Config to instantiate trainer. If ``config.train`` has
                ``trainer._target_``, its class is used. Otherwise, ``BaseTrainer`` is used.

        Returns:
            BaseTrainer: Built trainer.

        """
        train_dataset = instantiate(config.train.dataset.train)
        validation_dataset = instantiate(config.train.dataset.validation)

        train_loader = instantiate(config.train.dataloader.train, train_dataset)
        validation_loader = instantiate(config.train.dataloader.validation, validation_dataset)
        loaders = BaseDataLoaders(train_loader, validation_loader)

        model = instantiate_model(config.model)
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
            ddp_kwargs=config.train.ddp_kwargs,
        )
        criterion = instantiate_criterion(config.criterion)
        criterion = set_device(
            criterion,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
            ddp_kwargs=config.train.ddp_kwargs,
        )

        # TODO: support model and criterion by instantiate_optimizer.
        optimizer = instantiate_optimizer(config.optimizer, model)
        lr_scheduler = instantiate_lr_scheduler(config.lr_scheduler, optimizer)

        if hasattr(config.train, "clip_gradient"):
            # for backward compatibility
            grad_clipper = instantiate_grad_clipper(config.train.clip_gradient, model.parameters())
        else:
            grad_clipper = None

        if (
            hasattr(config.train, "trainer")
            and hasattr(config.train.trainer, "_target_")
            and config.train.trainer._target_ is not None
        ):
            # instantiate does not support config as keyword arguments.
            OmegaConf.update(
                config,
                "train.trainer",
                {"_partial_": True},
                merge=True,
                force_add=True,
            )
            trainer_cls = instantiate(
                config.train.trainer,
                loaders,
                model,
                optimizer,
                lr_scheduler=lr_scheduler,
                grad_clipper=grad_clipper,
                criterion=criterion,
            )
            trainer = trainer_cls(config=config)
        else:
            trainer = cls(
                loaders,
                model,
                optimizer,
                lr_scheduler=lr_scheduler,
                grad_clipper=grad_clipper,
                criterion=criterion,
                config=config,
            )

        return trainer

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        self.scaler = GradScaler(enabled=self.enable_amp)

        epochs = config.train.steps.epochs
        iterations = config.train.steps.iterations

        assert (epochs is not None and iterations is None) or (
            epochs is None and iterations is not None
        ), "Define either of config.train.epochs and config.train.iterations."

        if epochs is None:
            self.epochs = (iterations - 1) // len(self.loaders.train) + 1
            self.iterations = iterations
        else:
            self.epochs = epochs
            self.iterations = len(self.loaders.train) * epochs

        self.exp_dir = config.train.output.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        # Set git commit hash
        self.set_commit_hash()

        # Set loggder
        self.logger = get_logger(self.__class__.__name__, is_distributed=self.is_distributed)

        # Set tensorboard writer
        self.writer = get_summary_writer(
            log_dir=config.train.output.tensorboard_dir, is_distributed=self.is_distributed
        )

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        self.iteration_idx = 0
        self.best_loss = float("inf")
        self.epoch_idx = 0

        if config.train.resume.continue_from:
            continue_from = config.train.resume.continue_from
            self.logger.info(f"Resume training from: {continue_from}.")
            self.load_checkpoint(continue_from)

    def run(self) -> None:
        start_epoch_idx = self.epoch_idx
        criterion_names = self.criterion_names(self.config.criterion)

        for epoch_idx in range(start_epoch_idx, self.epochs):
            train_loss = self.train_one_epoch()

            if isinstance(self.optimizer, MovingAverageWrapper):
                self.logger.info("Set moving average model.")
                self.optimizer.set_moving_average_model()

            validation_loss = self.validate_one_epoch()
            self.infer_one_batch()

            if isinstance(self.optimizer, MovingAverageWrapper):
                self.logger.info("Remove moving average model.")
                self.optimizer.remove_moving_average_model()

            total_loss = self.display_loss(train_loss, validation_loss)

            for criterion_name in criterion_names:
                self.write_scalar_if_necessary(
                    f"{criterion_name} (epoch)/train",
                    train_loss[criterion_name],
                    global_step=self.epoch_idx + 1,
                )
                self.write_scalar_if_necessary(
                    f"{criterion_name} (epoch)/validation",
                    validation_loss[criterion_name],
                    global_step=self.epoch_idx + 1,
                )

            self.write_scalar_if_necessary(
                "total (epoch)/train",
                total_loss["train"],
                global_step=self.epoch_idx + 1,
            )
            self.write_scalar_if_necessary(
                "total (epoch)/validation",
                total_loss["validation"],
                global_step=self.epoch_idx + 1,
            )

            self.epoch_idx += 1

            if total_loss["validation"] < self.best_loss:
                self.best_loss = total_loss["validation"]

                if (
                    hasattr(self.config.train.output.save_checkpoint, "best_epoch")
                    and self.config.train.output.save_checkpoint.best_epoch
                ):
                    save_config = self.config.train.output.save_checkpoint.best_epoch
                    save_path = save_config.path.format(epoch=self.epoch_idx)
                    self.save_checkpoint_if_necessary(save_path)

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
        criterion_names = self.criterion_names(self.config.criterion)
        mean_metrics = {
            criterion_name: MeanMetric(device=self.device) for criterion_name in criterion_names
        }
        n_batch = 0
        n_remain = self.iteration_idx % len(self.loaders.train)

        self.set_epoch_if_necessary(self.epoch_idx)
        self.model.train()

        for named_data in self.loaders.train:
            if n_remain > 0:
                # When checkpoint is a specific iteration,
                # we have to skip the batches we've already treated.
                n_remain -= 1
                continue

            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(
                named_data, key_mapping=self.config.train.key_mapping.train
            )
            named_target = self.map_to_named_target(named_data)

            with autocast(enabled=self.enable_amp):
                output = self.model(**named_input)

                named_output = self.map_to_named_output(
                    output, key_mapping=self.config.train.key_mapping.train
                )
                named_estimated = self.map_to_named_estimated(named_output)

                total_loss = 0
                loss = {}

                for criterion_name in criterion_names:
                    weight = self.config.criterion[criterion_name].weight
                    loss[criterion_name] = self.criterion[criterion_name](
                        **named_estimated[criterion_name], **named_target[criterion_name]
                    )
                    total_loss = total_loss + weight * loss[criterion_name]

            for criterion_name in criterion_names:
                mean_metrics[criterion_name].update(loss[criterion_name].item())
                self.write_scalar_if_necessary(
                    f"{criterion_name} (iteration)/train",
                    loss[criterion_name].item(),
                    global_step=self.iteration_idx + 1,
                )

            self.write_scalar_if_necessary(
                "total (iteration)/train",
                total_loss,
                global_step=self.iteration_idx + 1,
            )

            self.write_train_duration_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
            )
            self.write_train_spectrogram_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
            )
            self.write_train_waveform_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
            )
            self.write_train_audio_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
            )
            self.write_train_image_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
            )

            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()
            self.unscale_optimizer_if_necessary()
            self.clip_gradient_if_necessary()
            self.optimizer_step(self.optimizer)
            self.scaler.update()

            if self.config.train.steps.lr_scheduler == "iteration":
                self.lr_scheduler_step(self.lr_scheduler, loss=loss)

            prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}"
            prompt += f", Iter {self.iteration_idx+1}/{self.iterations}]"
            s = ""

            for criterion_name in criterion_names:
                s += f"{criterion_name}: {loss[criterion_name]}, "

            s = f"{prompt} {total_loss.item()}, {s[:-2]}"

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

        train_loss = {}

        for criterion_name in criterion_names:
            loss = mean_metrics[criterion_name].compute()
            train_loss[criterion_name] = loss.item()

        if self.config.train.steps.lr_scheduler == "epoch":
            self.lr_scheduler_step(self.lr_scheduler, loss=train_loss)

        return train_loss

    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str, float]:
        """Validate model for one epoch."""
        criterion_names = self.criterion_names(self.config.criterion)
        mean_metrics = {
            criterion_name: MeanMetric(device=self.device) for criterion_name in criterion_names
        }
        n_batch = 0

        self.model.eval()

        for named_data in self.loaders.validation:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(
                named_data, key_mapping=self.config.train.key_mapping.validation
            )
            named_target = self.map_to_named_target(named_data)
            output = self.model(**named_input)
            named_output = self.map_to_named_output(
                output, key_mapping=self.config.train.key_mapping.validation
            )
            named_estimated = self.map_to_named_estimated(named_output)

            loss = {}

            for criterion_name in criterion_names:
                loss[criterion_name] = self.criterion[criterion_name](
                    **named_estimated[criterion_name], **named_target[criterion_name]
                )
                mean_metrics[criterion_name].update(loss[criterion_name].item())

            self.write_validation_duration_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_spectrogram_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_waveform_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_audio_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_validation_image_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )

            n_batch += 1

        validation_loss = {}

        for criterion_name in criterion_names:
            loss = mean_metrics[criterion_name].compute()
            validation_loss[criterion_name] = loss.item()

        return validation_loss

    @torch.no_grad()
    def infer_one_batch(self) -> None:
        """Inference using one batch."""
        if hasattr(self.config.train.key_mapping, "inference"):
            inference_key_mapping = self.config.train.key_mapping.inference
        elif hasattr(self.config.train.key_mapping, "validation"):
            inference_key_mapping = self.config.train.key_mapping.validation
        else:
            inference_key_mapping = self.config.train.key_mapping

        n_batch = 0

        self.model.eval()

        for named_data in self.loaders.validation:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=inference_key_mapping)

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(output, key_mapping=inference_key_mapping)

            self.write_inference_duration_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_spectrogram_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_waveform_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_audio_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )
            self.write_inference_image_if_necessary(
                named_output,
                named_data,
                config=self.config.train.record,
                batch_idx=n_batch,
            )

            n_batch += 1

            # Process only first batch.
            break

    def unscale_optimizer_if_necessary(self, enable: bool = True) -> None:
        """Unscale optimizer containing values if necessary.

        Args:
            enable (bool): If ``True``, ``self.scaler.unscale_`` is
                applied to ``self.optimizer``. This operation
                doesn't anything when ``self.scaler.is_enabled()`` is ``False``.

        """
        if enable:
            if isinstance(self.optimizer, MultiOptimizers):
                for optimizer in self.optimizer.optimizers.values():
                    self.scaler.unscale_(optimizer)
            else:
                self.scaler.unscale_(self.optimizer)

    def clip_gradient_if_necessary(self) -> None:
        """Clip gradient if self.grad_clipper is given."""
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
                self.grad_clipper = instantiate(clip_gradient_config, self.model.parameters())
                is_legacy = False
            elif _is_torch_clip_gradient(clip_gradient_fn):
                is_legacy = True
            else:
                raise ValueError("Invalid condition is detected.")

        if is_legacy:
            # for backward compatibility
            instantiate(clip_gradient_config, self.model.parameters())
        else:
            self.grad_clipper.step()

    def optimizer_step(self, optimizer: Optional[Optimizer] = None) -> None:
        if optimizer is None:
            optimizer = self.optimizer

        if isinstance(optimizer, MultiOptimizers):
            for _optimizer in optimizer.optimizers.values():
                self.optimizer_step(_optimizer)
        else:
            if isinstance(optimizer, ExponentialMovingAverageCodebookOptimizer):
                optimizer.step()
            else:
                self.scaler.step(optimizer)

    def lr_scheduler_step(
        self,
        lr_scheduler: Optional[_LRScheduler] = None,
        loss: Optional[Dict[str, Union[torch.Tensor, float]]] = None,
    ) -> None:
        """Call .step of learning rate scheduler.

        Args:
            lr_scheduler (_LRScheduler): Learning rate scheduler.

        """
        if lr_scheduler is None:
            lr_scheduler = self.lr_scheduler

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            if loss is None:
                raise ValueError("ReduceLROnPlateau requires loss to update.")

            total_loss = 0
            criterion_names = self.criterion_names(self.config.criterion)

            for criterion_name in criterion_names:
                weight = self.config.criterion[criterion_name].weight
                loss = loss[criterion_name]
                total_loss += weight * loss

            lr_scheduler.step(total_loss)
        else:
            lr_scheduler.step()

    def display_loss(
        self, train_loss: Dict[str, float], validation_loss: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        train_key = "train"
        validation_key = "validation"
        total_loss = {train_key: 0, validation_key: 0}
        criterion_names = self.criterion_names(self.config.criterion)

        prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}] ({train_key})"
        s = ""

        for criterion_name in criterion_names:
            weight = self.config.criterion[criterion_name].weight
            loss = train_loss[criterion_name]
            total_loss[train_key] = total_loss[train_key] + weight * loss
            s += f"{criterion_name}: {loss}, "

        s = f"{prompt} {total_loss[train_key]}, {s[:-2]}"
        self.logger.info(s)

        prompt = f"[Epoch {self.epoch_idx+1}/{self.epochs}] ({validation_key})"
        s = ""

        for criterion_name in criterion_names:
            weight = self.config.criterion[criterion_name].weight
            loss = validation_loss[criterion_name]
            total_loss[validation_key] = total_loss[validation_key] + weight * loss
            s += f"{criterion_name}: {loss}, "

        s = f"{prompt} {total_loss[validation_key]}, {s[:-2]}"
        self.logger.info(s)

        return total_loss

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        # model
        self.unwrapped_model.load_state_dict(state_dict["model"])

        # optimizer
        self.optimizer.load_state_dict(state_dict["optimizer"])

        # learning rate scheduler
        if self.lr_scheduler is None:
            assert state_dict["lr_scheduler"] is None
        else:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])

        # gradient scaler
        self.scaler.load_state_dict(state_dict["scaler"])

        self.iteration_idx = state_dict["iteration_idx"]
        self.best_loss = state_dict["best_loss"]
        self.epoch_idx = self.iteration_idx // len(self.loaders.train)

    @run_only_master_rank()
    def save_checkpoint_if_necessary(self, save_path: str) -> None:
        self.save_checkpoint(save_path)

    def save_checkpoint(self, save_path: str) -> None:
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)

        state_dict = {}

        # model
        state_dict["model"] = self.unwrapped_model.state_dict()

        # optimizer
        state_dict["optimizer"] = self.optimizer.state_dict()

        # learning rate scheduler
        if self.lr_scheduler is None:
            state_dict["lr_scheduler"] = None
        else:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()

        # gradient scaler
        state_dict["scaler"] = self.scaler.state_dict()

        state_dict["iteration_idx"] = self.iteration_idx
        state_dict["best_loss"] = self.best_loss
        state_dict["resolved_config"] = OmegaConf.to_container(self.config, resolve=True)

        if isinstance(self.optimizer, MovingAverageWrapper):
            # Save state dict of moving averaged model
            self.optimizer.set_moving_average_model()
            state_dict["moving_average_model"] = self.unwrapped_model.state_dict()
            self.optimizer.remove_moving_average_model()

        # Store metadata
        module_name, class_name = self.__module__, self.__class__.__name__
        class_name = class_name if module_name is None else f"{module_name}.{class_name}"
        commit_hash = self.commit_hash
        state_dict["_metadata"] = {
            "version": _version,
            "driver": class_name,
            "commit_hash": commit_hash,
        }

        torch.save(state_dict, save_path)

        s = f"Save model: {save_path}."
        self.logger.info(s)

    def write_train_duration_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
    ) -> None:
        """Write duration to tensorboard for training.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "duration"):
            duration_config = config.duration.iteration
            global_step = self.iteration_idx + 1

            if duration_config is not None and global_step % duration_config.every == 0:
                self.write_duration_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=duration_config.sample_size,
                    key_mapping=duration_config.key_mapping,
                    transforms=duration_config.transforms,
                    global_step=global_step,
                )

    def write_train_spectrogram_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
    ) -> None:
        """Write spectrogram to tensorboard for training.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "spectrogram"):
            spectrogram_config = config.spectrogram.iteration
            global_step = self.iteration_idx + 1

            if spectrogram_config is not None and global_step % spectrogram_config.every == 0:
                self.write_spectrogram_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=spectrogram_config.sample_size,
                    key_mapping=spectrogram_config.key_mapping,
                    transforms=spectrogram_config.transforms,
                    global_step=global_step,
                )

    def write_train_waveform_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
    ) -> None:
        """Write waveform to tensorboard for training.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "spectrogram"):
            spectrogram_config = config.spectrogram.iteration
            global_step = self.iteration_idx + 1

            if spectrogram_config is not None and global_step % spectrogram_config.every == 0:
                self.write_spectrogram_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=spectrogram_config.sample_size,
                    key_mapping=spectrogram_config.key_mapping,
                    transforms=spectrogram_config.transforms,
                    global_step=global_step,
                )

    def write_train_audio_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
    ) -> None:
        """Write audio to tensorboard for training.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "audio"):
            audio_config = config.audio.iteration
            global_step = self.iteration_idx + 1

            if audio_config is not None and global_step % audio_config.every == 0:
                self.write_audio_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=audio_config.sample_size,
                    key_mapping=audio_config.key_mapping,
                    transforms=audio_config.transforms,
                    global_step=global_step,
                    sample_rate=audio_config.sample_rate,
                )

    def write_train_image_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
    ) -> None:
        """Write image to tensorboard for training.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "image"):
            image_config = config.image.iteration
            global_step = self.iteration_idx + 1

            if image_config is not None and global_step % image_config.every == 0:
                self.write_image_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=image_config.sample_size,
                    key_mapping=image_config.key_mapping,
                    transforms=image_config.transforms,
                    global_step=global_step,
                )

    def write_validation_duration_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write duration to tensorboard for validation.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "duration") and batch_idx < 1:
            duration_config = config.duration.epoch
            global_step = self.epoch_idx + 1

            if duration_config is not None and global_step % duration_config.every == 0:
                if hasattr(duration_config.key_mapping, "validation"):
                    key_mapping = duration_config.key_mapping.validation
                else:
                    key_mapping = duration_config.key_mapping

                if hasattr(duration_config.key_mapping, "validation"):
                    transforms = duration_config.transforms.validation
                else:
                    transforms = duration_config.transforms

                self.write_duration_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=duration_config.sample_size,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    global_step=global_step,
                )

    def write_validation_spectrogram_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write spectrogram to tensorboard for validation.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "spectrogram") and batch_idx < 1:
            spectrogram_config = config.spectrogram.epoch
            global_step = self.epoch_idx + 1

            if spectrogram_config is not None and global_step % spectrogram_config.every == 0:
                if hasattr(spectrogram_config.key_mapping, "validation"):
                    key_mapping = spectrogram_config.key_mapping.validation
                else:
                    key_mapping = spectrogram_config.key_mapping

                if hasattr(spectrogram_config.key_mapping, "validation"):
                    transforms = spectrogram_config.transforms.validation
                else:
                    transforms = spectrogram_config.transforms

                self.write_spectrogram_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=spectrogram_config.sample_size,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    global_step=global_step,
                )

    def write_validation_waveform_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write waveform to tensorboard for validation.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "waveform") and batch_idx < 1:
            waveform_config = config.waveform.epoch
            global_step = self.epoch_idx + 1

            if waveform_config is not None and global_step % waveform_config.every == 0:
                if hasattr(waveform_config.key_mapping, "validation"):
                    key_mapping = waveform_config.key_mapping.validation
                else:
                    key_mapping = waveform_config.key_mapping

                if hasattr(waveform_config.transforms, "validation"):
                    transforms = waveform_config.transforms.validation
                else:
                    transforms = waveform_config.transforms

                self.write_waveform_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=waveform_config.sample_size,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    global_step=global_step,
                )

    def write_validation_audio_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write audio to tensorboard for validation.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "audio") and batch_idx < 1:
            audio_config = config.audio.epoch
            global_step = self.epoch_idx + 1

            if audio_config is not None and global_step % audio_config.every == 0:
                if hasattr(audio_config.key_mapping, "validation"):
                    key_mapping = audio_config.key_mapping.validation
                else:
                    key_mapping = audio_config.key_mapping

                if hasattr(audio_config.transforms, "validation"):
                    transforms = audio_config.transforms.validation
                else:
                    transforms = audio_config.transforms

                self.write_audio_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=audio_config.sample_size,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    global_step=global_step,
                    sample_rate=audio_config.sample_rate,
                )

    def write_validation_image_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write image to tensorboard for validation.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "image") and batch_idx < 1:
            image_config = config.image.epoch
            global_step = self.epoch_idx + 1

            if image_config is not None and global_step % image_config.every == 0:
                if hasattr(image_config.key_mapping, "validation"):
                    key_mapping = image_config.key_mapping.validation
                else:
                    key_mapping = image_config.key_mapping

                if hasattr(image_config.transforms, "validation"):
                    transforms = image_config.transforms.validation
                else:
                    transforms = image_config.transforms

                self.write_image_if_necessary(
                    named_output,
                    named_reference,
                    sample_size=image_config.sample_size,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    global_step=global_step,
                )

    def write_inference_duration_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write duration to tensorboard for inference.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "duration") and batch_idx < 1:
            duration_config = config.duration.epoch
            global_step = self.epoch_idx + 1

            if duration_config is not None and global_step % duration_config.every == 0:
                if hasattr(duration_config.key_mapping, "inference"):
                    key_mapping = duration_config.key_mapping.inference

                    if hasattr(duration_config.transforms, "inference"):
                        transforms = duration_config.transforms.inference
                    elif hasattr(duration_config.transforms, "validation"):
                        transforms = duration_config.transforms.validation
                    else:
                        transforms = duration_config.transforms

                    self.write_duration_if_necessary(
                        named_output,
                        named_reference,
                        sample_size=duration_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

    def write_inference_spectrogram_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write spectrogram to tensorboard for inference.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "spectrogram") and batch_idx < 1:
            spectrogram_config = config.spectrogram.epoch
            global_step = self.epoch_idx + 1

            if spectrogram_config is not None and global_step % spectrogram_config.every == 0:
                if hasattr(spectrogram_config.key_mapping, "inference"):
                    key_mapping = spectrogram_config.key_mapping.inference

                    if hasattr(spectrogram_config.transforms, "inference"):
                        transforms = spectrogram_config.transforms.inference
                    elif hasattr(spectrogram_config.transforms, "validation"):
                        transforms = spectrogram_config.transforms.validation
                    else:
                        transforms = spectrogram_config.transforms

                    self.write_spectrogram_if_necessary(
                        named_output,
                        named_reference,
                        sample_size=spectrogram_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

    def write_inference_waveform_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write waveform to tensorboard for inference.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "waveform") and batch_idx < 1:
            waveform_config = config.waveform.epoch
            global_step = self.epoch_idx + 1

            if waveform_config is not None and global_step % waveform_config.every == 0:
                if hasattr(waveform_config.key_mapping, "inference"):
                    key_mapping = waveform_config.key_mapping.inference

                    if hasattr(waveform_config.transforms, "inference"):
                        transforms = waveform_config.transforms.inference
                    elif hasattr(waveform_config.transforms, "validation"):
                        transforms = waveform_config.transforms.validation
                    else:
                        transforms = waveform_config.transforms

                    self.write_waveform_if_necessary(
                        named_output,
                        named_reference,
                        sample_size=waveform_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

    def write_inference_audio_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write audio to tensorboard for inference.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "audio") and batch_idx < 1:
            audio_config = config.audio.epoch
            global_step = self.epoch_idx + 1

            if audio_config is not None and global_step % audio_config.every == 0:
                if hasattr(audio_config.key_mapping, "inference"):
                    key_mapping = audio_config.key_mapping.inference

                    if hasattr(audio_config.transforms, "inference"):
                        transforms = audio_config.transforms.inference
                    elif hasattr(audio_config.transforms, "validation"):
                        transforms = audio_config.transforms.validation
                    else:
                        transforms = audio_config.transforms

                    self.write_audio_if_necessary(
                        named_output,
                        named_reference,
                        sample_size=audio_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                        sample_rate=audio_config.sample_rate,
                    )

    def write_inference_image_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        config: DictConfig = None,
        batch_idx: int = 0,
    ) -> None:
        """Write image to tensorboard for inference.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            config (DictConfig, optional): Config to write out to tensorboard.
            batch_idx (int): Batch index.

        """
        if config is None:
            config = self.config.train.record

        if hasattr(config, "image") and batch_idx < 1:
            image_config = config.image.epoch
            global_step = self.epoch_idx + 1

            if image_config is not None and global_step % image_config.every == 0:
                if hasattr(image_config.key_mapping, "inference"):
                    key_mapping = image_config.key_mapping.inference

                    if hasattr(image_config.transforms, "inference"):
                        transforms = image_config.transforms.inference
                    elif hasattr(image_config.transforms, "validation"):
                        transforms = image_config.transforms.validation
                    else:
                        transforms = image_config.transforms

                    self.write_image_if_necessary(
                        named_output,
                        named_reference,
                        sample_size=image_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

    @run_only_master_rank()
    def write_scalar_if_necessary(self, tag: Any, scalar_value: Any, global_step: Any) -> None:
        self.writer.add_scalar(
            tag,
            scalar_value,
            global_step=global_step,
        )

    @run_only_master_rank()
    def write_duration_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        sample_size: int = 1,
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
        global_step: int = 1,
    ) -> None:
        """Write duration as figure to tensorboard.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.
            global_step (int): Step value to record.

        """
        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            for key, tag in key_mapping.output.items():
                for sample_idx, duration in enumerate(named_output[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            duration = transform(duration)

                    self.write_duration(
                        tag.format(number=sample_idx + 1), duration, global_step=global_step
                    )

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, tag in key_mapping.reference.items():
                for sample_idx, duration in enumerate(named_reference[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            duration = transform(duration)

                    self.write_duration(
                        tag.format(number=sample_idx + 1), duration, global_step=global_step
                    )

    def write_duration(self, tag: str, duration: torch.Tensor, global_step: Any = 1) -> None:
        """Write out duration to tensorboard as 2D alignment map.

        Args:
            duration (torch.Tensor): (src_length,) or (tgt_length, src_length).

        """
        assert (
            duration.dim() == 1 or duration.dim() == 2
        ), f"duration is expected to be 1D or 2D tesor, but given as {duration.dim()}D tensor."

        if duration.dim() == 1:
            eye = torch.eye(duration.size(-1), dtype=torch.long, device=duration.device)

            # add pseudo batch dimension
            eye = eye.unsqueeze(dim=0)
            duration = duration.unsqueeze(dim=0)
            alignment = expand_by_duration(eye, duration)

            # remove pseudo batch dimension
            alignment = alignment.squeeze(dim=0)
            alignment = alignment.permute(1, 0).contiguous()
            tgt_length = torch.count_nonzero(alignment, dim=1)
            tgt_length = torch.count_nonzero(tgt_length).item()
            src_length = torch.count_nonzero(alignment, dim=0)
            src_length = torch.count_nonzero(src_length).item()
        else:
            # TODO: permute dims if necessary
            raise NotImplementedError("2D input is not supported now.")

        alignment = alignment.detach().cpu()
        max_tgt_length, max_src_length = alignment.size()
        alignment = F.pad(
            alignment, (0, src_length - max_src_length, 0, tgt_length - max_tgt_length)
        )

        fig, axis = plt.subplots(figsize=(16, 10))
        im = axis.pcolormesh(alignment)
        fig.colorbar(im, ax=axis, fraction=0.05)
        fig.tight_layout()

        self.writer.add_figure(tag, fig, global_step=global_step)

    @run_only_master_rank()
    def write_spectrogram_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        sample_size: int = 1,
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
        global_step: int = 1,
    ) -> None:
        """Write spectrogram as figure to tensorboard.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.
            global_step (int): Step value to record.

        """
        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            for key, tag in key_mapping.output.items():
                for sample_idx, spectrogram in enumerate(named_output[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            spectrogram = transform(spectrogram)

                    self.write_spectrogram(
                        tag.format(number=sample_idx + 1), spectrogram, global_step=global_step
                    )

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, tag in key_mapping.reference.items():
                for sample_idx, spectrogram in enumerate(named_reference[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            spectrogram = transform(spectrogram)

                    self.write_spectrogram(
                        tag.format(number=sample_idx + 1), spectrogram, global_step=global_step
                    )

    def write_spectrogram(self, tag: str, spectrogram: torch.Tensor, global_step: Any = 1) -> None:
        assert (
            spectrogram.dim() == 2
        ), f"spectrogram is expected to be 2D tesor, but given as {spectrogram.dim()}D tensor."

        spectrogram = spectrogram.detach().cpu()

        fig, axis = plt.subplots(figsize=(16, 10))
        im = axis.pcolormesh(spectrogram)
        fig.colorbar(im, ax=axis, fraction=0.05)
        fig.tight_layout()

        self.writer.add_figure(tag, fig, global_step=global_step)

    @run_only_master_rank()
    def write_waveform_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        sample_size: int = 1,
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
        global_step: int = 1,
    ) -> None:
        """Write waveform as figure to tensorboard.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.
            global_step (int): Step value to record.

        """
        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, tag in key_mapping.output.items():
                for sample_idx, waveform in enumerate(named_output[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            waveform = transform(waveform)

                    self.write_waveform(
                        tag.format(number=sample_idx + 1), waveform, global_step=global_step
                    )

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, tag in key_mapping.reference.items():
                for sample_idx, waveform in enumerate(named_reference[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            waveform = transform(waveform)

                    self.write_waveform(
                        tag.format(number=sample_idx + 1), waveform, global_step=global_step
                    )

    def write_waveform(self, tag: str, waveform: torch.Tensor, global_step: Any = 1) -> None:
        if waveform.dim() == 2:
            assert (
                waveform.size(0) == 1
            ), "First dimension is expected to be 1, but given {}.".format(waveform.size(0))

            waveform = waveform.squeeze(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(
                "waveform is expected to be 1D tesor, but given as {}D tensor.".format(
                    waveform.dim()
                )
            )

        waveform = waveform.detach().cpu()

        fig, axis = plt.subplots(figsize=(16, 10))
        axis.plot(waveform)
        fig.tight_layout()

        self.writer.add_figure(tag, fig, global_step=global_step)

    @run_only_master_rank()
    def write_audio_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        sample_size: int = 1,
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
        global_step: int = 1,
        sample_rate: int = 44100,
    ) -> None:
        """Write audio to tensorboard.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.
            global_step (int): Step value to record.
            sample_rate (int): Sampling rate of audio. Default: 44100.

        """
        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, tag in key_mapping.output.items():
                for sample_idx, waveform in enumerate(named_output[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            waveform = transform(waveform)

                    self.write_audio(
                        tag.format(number=sample_idx + 1),
                        waveform,
                        global_step=global_step,
                        sample_rate=sample_rate,
                    )

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, tag in key_mapping.reference.items():
                for sample_idx, waveform in enumerate(named_reference[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            waveform = transform(waveform)

                    self.write_audio(
                        tag.format(number=sample_idx + 1),
                        waveform,
                        global_step=global_step,
                        sample_rate=sample_rate,
                    )

    def write_audio(
        self, tag: str, waveform: torch.Tensor, global_step: Any = 1, sample_rate: int = 44100
    ) -> None:
        if waveform.dim() == 2:
            assert (
                waveform.size(0) == 1
            ), "First dimension is expected to be 1, but given {}.".format(waveform.size(0))

            waveform = waveform.squeeze(dim=0)
        elif waveform.dim() != 1:
            raise ValueError(
                "waveform is expected to be 1D tesor, but given as {}D tensor.".format(
                    waveform.dim()
                )
            )

        waveform = waveform.detach().cpu()
        self.writer.add_audio(tag, waveform, global_step=global_step, sample_rate=sample_rate)

    @run_only_master_rank()
    def write_image_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]] = None,
        named_reference: Optional[Dict[str, torch.Tensor]] = None,
        sample_size: int = 1,
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
        global_step: int = 1,
    ) -> None:
        """Write image to tensorboard.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.
            global_step (int): Step value to record.

        """
        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, tag in key_mapping.output.items():
                for sample_idx, image in enumerate(named_output[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output:
                            transform = instantiate(transforms.output[key])
                            image = transform(image)

                    self.write_image(
                        tag.format(number=sample_idx + 1), image, global_step=global_step
                    )

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, tag in key_mapping.reference.items():
                for sample_idx, image in enumerate(named_reference[key]):
                    if sample_idx >= sample_size:
                        break

                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference:
                            transform = instantiate(transforms.reference[key])
                            image = transform(image)

                    self.write_image(
                        tag.format(number=sample_idx + 1), image, global_step=global_step
                    )

    def write_image(self, tag: str, image: torch.Tensor, global_step: Any = 1) -> None:
        assert image.dim() in [
            2,
            3,
        ], f"waveform is expected to be 2 or 3D tesor, but given as {image.dim()}D tensor."

        image = image.detach().cpu()

        if image.dim() == 2:
            image = image.unsqueeze(dim=0)

        self.writer.add_image(tag, image, global_step=global_step)

    def set_epoch_if_necessary(self, epoch: int) -> None:
        sampler = None

        if self.loaders.train.sampler is not None:
            sampler = self.loaders.train.sampler
        elif self.loaders.train.batch_sampler is not None:
            sampler = self.loaders.train.batch_sampler

        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def set_commit_hash(self) -> None:
        """Set git commit hash."""
        try:
            completed_process = subprocess.run(
                "git rev-parse HEAD", shell=True, check=True, capture_output=True
            )
            commit_hash = completed_process.stdout.decode().strip()
        except subprocess.CalledProcessError:
            commit_hash = None
            warnings.warn("The system is not managed by git.", UserWarning)

        self.commit_hash = commit_hash


class BaseGenerator(BaseDriver):
    def __init__(
        self,
        loader: DataLoader,
        model: nn.Module,
        config: DictConfig = None,
    ) -> None:
        self.loader = loader
        self.model = model

        self.config = config

        self._reset(config)

    @classmethod
    def build_from_config(cls, config: DictConfig) -> "BaseTrainer":
        """Instantiate Generator from config.

        Args:
            config (DictConfig): Config to instantiate generator. If ``config.test`` has
                ``generator._target_``, its class is used. Otherwise, ``BaseGenerator`` is used.

        Returns:
            BaseGenerator: Built generator.

        """
        test_dataset = instantiate(config.test.dataset.test)
        test_loader = instantiate(config.test.dataloader.test, test_dataset)
        model = instantiate_model(config.test.checkpoint)
        model = set_device(
            model,
            accelerator=config.system.accelerator,
            is_distributed=config.system.distributed.enable,
        )

        if (
            hasattr(config.test, "generator")
            and hasattr(config.test.generator, "_target_")
            and config.test.generator._target_ is not None
        ):
            # instantiate does not support config as keyword arguments.
            OmegaConf.update(
                config,
                "test.generator",
                {"_partial_": True},
                merge=True,
                force_add=True,
            )
            generator_cls = instantiate(
                config.test.generator,
                test_loader,
                model,
            )
            generator = generator_cls(config=config)
        else:
            generator = cls(
                test_loader,
                model,
                config=config,
            )

        return generator

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        self.exp_dir = config.test.output.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        self.inference_dir = config.test.output.inference_dir
        os.makedirs(self.inference_dir, exist_ok=True)

        # Set loggder
        self.logger = get_logger(
            self.__class__.__name__,
            is_distributed=self.is_distributed,
        )

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        checkpoint = config.test.checkpoint

        self.logger.info(f"Load weights of model: {checkpoint}.")
        self.load_checkpoint(checkpoint)

        self.remove_weight_norm_if_necessary()

    @torch.no_grad()
    def run(self) -> None:
        test_config = self.config.test
        key_mapping = test_config.key_mapping.inference

        self.model.eval()

        for named_data in self.loader:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=key_mapping)
            named_identifier = self.map_to_named_identifier(named_data, key_mapping=key_mapping)

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(output, key_mapping=key_mapping)

            self.save_inference_torch_dump_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )
            self.save_inference_audio_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )
            self.save_inference_spectrogram_if_necessary(
                named_output,
                named_data,
                named_identifier,
                config=test_config.output,
            )

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])

    def save_inference_torch_dump_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        config: DictConfig = None,
    ) -> None:
        if config is None:
            config = config.test.output

        if hasattr(config, "torch_dump"):
            torch_dump_config = config.torch_dump

            if torch_dump_config is not None:
                if hasattr(torch_dump_config.key_mapping, "inference"):
                    key_mapping = torch_dump_config.key_mapping.inference
                elif hasattr(torch_dump_config.key_mapping, "test"):
                    key_mapping = torch_dump_config.key_mapping.test
                else:
                    key_mapping = torch_dump_config.key_mapping

                if hasattr(torch_dump_config.key_mapping, "inference"):
                    transforms = torch_dump_config.transforms.inference
                elif hasattr(torch_dump_config.key_mapping, "test"):
                    transforms = torch_dump_config.transforms.test
                else:
                    transforms = torch_dump_config.transforms

                self.save_torch_dump_if_necessary(
                    named_output,
                    named_reference,
                    named_identifier,
                    key_mapping=key_mapping,
                    transforms=transforms,
                )

    def save_inference_audio_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        config: DictConfig = None,
    ) -> None:
        if config is None:
            config = config.test.output

        if hasattr(config, "audio"):
            audio_config = config.audio

            if audio_config is not None:
                if hasattr(audio_config.key_mapping, "inference"):
                    key_mapping = audio_config.key_mapping.inference
                elif hasattr(audio_config.key_mapping, "test"):
                    key_mapping = audio_config.key_mapping.test
                else:
                    key_mapping = audio_config.key_mapping

                if hasattr(audio_config.key_mapping, "inference"):
                    transforms = audio_config.transforms.inference
                elif hasattr(audio_config.key_mapping, "test"):
                    transforms = audio_config.transforms.test
                else:
                    transforms = audio_config.transforms

                self.save_audio_if_necessary(
                    named_output,
                    named_reference,
                    named_identifier,
                    key_mapping=key_mapping,
                    transforms=transforms,
                    sample_rate=audio_config.sample_rate,
                )

    def save_inference_spectrogram_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        config: DictConfig = None,
    ) -> None:
        if config is None:
            config = config.test.output

        if hasattr(config, "spectrogram"):
            spectrogram_config = config.spectrogram

            if spectrogram_config is not None:
                if hasattr(spectrogram_config.key_mapping, "inference"):
                    key_mapping = spectrogram_config.key_mapping.inference
                elif hasattr(spectrogram_config.key_mapping, "test"):
                    key_mapping = spectrogram_config.key_mapping.test
                else:
                    key_mapping = spectrogram_config.key_mapping

                if hasattr(spectrogram_config.key_mapping, "inference"):
                    transforms = spectrogram_config.transforms.inference
                elif hasattr(spectrogram_config.key_mapping, "test"):
                    transforms = spectrogram_config.transforms.test
                else:
                    transforms = spectrogram_config.transforms

                self.save_spectrogram_if_necessary(
                    named_output,
                    named_reference,
                    named_identifier,
                    key_mapping=key_mapping,
                    transforms=transforms,
                )

    @run_only_master_rank()
    def save_torch_dump_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        key_mapping: DictConfig = None,
        transforms: DictConfig = None,
    ) -> None:
        identifier_keys = named_identifier.keys()

        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, filename in key_mapping.output.items():
                for sample_idx, output in enumerate(named_output[key]):
                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            output = transform(output)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_torch_dump(output, path)

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, filename in key_mapping.reference.items():
                for sample_idx, output in enumerate(named_reference[key]):
                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            output = transform(output)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_torch_dump(output, path)

    def save_torch_dump(
        self,
        obj: Any,
        path: str,
    ) -> None:
        """Save torch dump object via torch.save.

        Args:
            kwargs: Keyword arguments given to ``torch.save``.

        .. note::

            If ``obj`` is instance of ``torch.Tensor``, ``.detach().cpu()`` is called.

        """
        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        torch.save(obj, path)

    @run_only_master_rank()
    def save_audio_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        key_mapping: DictConfig = None,
        transforms: DictConfig = None,
        sample_rate: int = 44100,
    ) -> None:
        identifier_keys = named_identifier.keys()

        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, filename in key_mapping.output.items():
                for sample_idx, waveform in enumerate(named_output[key]):
                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            waveform = transform(waveform)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_audio(path, waveform, sample_rate=sample_rate)

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, filename in key_mapping.reference.items():
                for sample_idx, waveform in enumerate(named_reference[key]):
                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            waveform = transform(waveform)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_audio(path, waveform, sample_rate=sample_rate)

    def save_audio(
        self,
        path: str,
        waveform: torch.Tensor,
        sample_rate: int = 44100,
        **kwargs,
    ) -> None:
        """Save audio via torchaudio.save.

        Args:
            kwargs: Keyword arguments given to ``torchaudio.save``.

        """
        assert waveform.dim() in [
            1,
            2,
        ], f"waveform is expected to be 1 or 2D tesor, but given as {waveform.dim()}D tensor."

        waveform = waveform.detach().cpu()

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(dim=0)

        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        kwargs = {}

        is_torchaudio_lt_2_1 = version.parse(torchaudio.__version__) < version.parse("2.1")

        if is_torchaudio_lt_2_1:
            audio_backend = torchaudio.get_audio_backend()
        else:
            audio_backends = torchaudio.list_audio_backends()

            if len(audio_backends) == 0:
                # torchaudio.list_audio_backends sometimes returns [].
                audio_backend = None
            else:
                if "ffmpeg" in audio_backends:
                    # default audio backend in torchaudio
                    audio_backend = "ffmpeg"
                elif "sox" in audio_backends:
                    audio_backend = "sox"
                elif "soundfile" in audio_backends:
                    audio_backend = "soundfile"
                else:
                    raise ValueError(
                        "Available audio backends are not found from {}.".format(audio_backends)
                    )

        if audio_backend in ["sox", "sox_io"]:
            valid_kwargs = {"bits_per_sample": 16}
        elif audio_backend in [None, "soundfile", "ffmpeg"]:
            valid_kwargs = {}
        else:
            raise ValueError("Invalid audio backend {} is detected.".format(audio_backend))

        if not is_torchaudio_lt_2_1 and audio_backend is not None:
            valid_kwargs["backend"] = audio_backend

        invalid_keys = set(kwargs) - set(valid_kwargs.keys())

        if invalid_keys != set():
            raise ValueError("Invalid keywords {} are given.".format(invalid_keys))

        for valid_key, default_value in valid_kwargs.items():
            if valid_key not in kwargs:
                kwargs[valid_key] = default_value

        torchaudio.save(
            path,
            waveform,
            sample_rate=sample_rate,
            **kwargs,
        )

    @run_only_master_rank()
    def save_spectrogram_if_necessary(
        self,
        named_output: Optional[Dict[str, torch.Tensor]],
        named_reference: Optional[Dict[str, torch.Tensor]],
        named_identifier: Dict[str, List[str]],
        key_mapping: DictConfig = None,
        transforms: Optional[DictConfig] = None,
    ) -> None:
        """Save spectrogram.

        Args:
            named_output (dict, optional): Estimated data.
            named_reference (dict, optional): Target data.
            named_identifier (dict, optional): Identifier to save sample.
            key_mapping (DictConfig): Config to map data.
            transforms (DictConfig, optional): Config to transform data.

        """
        identifier_keys = named_identifier.keys()

        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            for key, filename in key_mapping.output.items():
                for sample_idx, image in enumerate(named_output[key]):
                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = instantiate(transforms.output[key])
                            image = transform(image)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_spectrogram(path, image)

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, filename in key_mapping.reference.items():
                for sample_idx, image in enumerate(named_reference[key]):
                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = instantiate(transforms.reference[key])
                            image = transform(image)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][sample_idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_spectrogram(path, image)

    def save_spectrogram(self, path: str, spectrogram: torch.Tensor) -> None:
        """Save spectrogram using matplotlib."""
        assert (
            spectrogram.dim() == 2
        ), f"spectrogram is expected to be 2D tesor, but given as {spectrogram.dim()}D tensor."

        spectrogram = spectrogram.detach().cpu()

        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        fig, axis = plt.subplots(figsize=(16, 10))
        im = axis.pcolormesh(spectrogram)
        fig.colorbar(im, ax=axis, fraction=0.05)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close()

    def remove_weight_norm_if_necessary(self) -> None:
        """Remove weight normalization from self.model by calling self.model.remove_weight_norm()
        or self.model.remove_weight_norm_().
        """
        if not hasattr(self.config.test, "remove_weight_norm"):
            return

        if not self.config.test.remove_weight_norm:
            return

        if hasattr(self.unwrapped_model, "remove_weight_norm") and callable(
            self.unwrapped_model.remove_weight_norm
        ):
            self.unwrapped_model.remove_weight_norm()

        if hasattr(self.unwrapped_model, "remove_weight_norm_") and callable(
            self.unwrapped_model.remove_weight_norm_
        ):
            self.unwrapped_model.remove_weight_norm_()


def _is_torch_clip_gradient(cls: type) -> bool:
    for clip_gradient_fn in TORCH_CLIP_GRAD_FN:
        mod_name, var_name = clip_gradient_fn.rsplit(".", maxsplit=1)

        if cls is getattr(importlib.import_module(mod_name), var_name):
            return True

    return False


def _is_audyn_clip_gradient(cls: type) -> bool:
    if cls is GradClipper:
        return True

    return False
