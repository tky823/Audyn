from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...metrics import MeanMetric
from ..clip_grad import GradClipper
from ..data import BaseDataLoaders
from ..hydra.utils import instantiate_model
from ..model import set_device, unwrap
from .base import BaseTrainer


class TextToFeatTrainer(BaseTrainer):
    """Trainer for text-to-feat model."""

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
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            grad_clipper=grad_clipper,
            criterion=criterion,
            config=config,
        )

    def _reset(self, config: DictConfig) -> None:
        self.set_feat_to_wave_if_necessary(config=config)
        super()._reset(config=config)

    def set_feat_to_wave_if_necessary(self, config: Optional[DictConfig] = None) -> None:
        """Set feat-to-wave model if specified."""
        if config is None:
            config = self.config

        system_config = config.system
        train_config = config.train

        feat_to_wave = None
        transform_middle = None
        use_pretrained_feat_to_wave = False

        if hasattr(train_config, "pretrained_feat_to_wave"):
            pretrained_config = train_config.pretrained_feat_to_wave

            if hasattr(pretrained_config, "path") and pretrained_config.path:
                use_pretrained_feat_to_wave = True

        if use_pretrained_feat_to_wave:
            state_dict = torch.load(
                pretrained_config.path, map_location=lambda storage, loc: storage
            )
            feat_to_wave_config = OmegaConf.create(state_dict["resolved_config"])

            if "_target_" in feat_to_wave_config.model:
                feat_to_wave = instantiate_model(feat_to_wave_config.model)
                feat_to_wave = set_device(
                    feat_to_wave,
                    accelerator=system_config.accelerator,
                    is_distributed=system_config.distributed.enable,
                    ddp_kwargs=config.train.ddp_kwargs,
                )
                unwrapped_feat_to_wave = unwrap(feat_to_wave)
                unwrapped_feat_to_wave.load_state_dict(state_dict["model"])
            elif "generator" in feat_to_wave_config.model:
                # generator of GAN
                feat_to_wave = instantiate_model(feat_to_wave_config.model.generator)
                feat_to_wave = set_device(
                    feat_to_wave,
                    accelerator=system_config.accelerator,
                    is_distributed=system_config.distributed.enable,
                    ddp_kwargs=config.train.ddp_kwargs,
                )
                unwrapped_feat_to_wave = unwrap(feat_to_wave)
                unwrapped_feat_to_wave.load_state_dict(state_dict["model"]["generator"])
            else:
                raise ValueError("Given config type is not supported now.")

            if hasattr(pretrained_config, "transform_middle"):
                transform_middle = pretrained_config.transform_middle

                if transform_middle is not None:
                    if feat_to_wave is None:
                        raise ValueError(
                            "transform_middle is defined, but feat_to_wave is not defined."
                        )

                    transform_middle = instantiate_model(transform_middle)
                    transform_middle = set_device(
                        transform_middle,
                        accelerator=system_config.accelerator,
                        is_distributed=system_config.distributed.enable,
                        ddp_kwargs=config.train.ddp_kwargs,
                    )

        self.feat_to_wave = feat_to_wave
        self.transform_middle = transform_middle

    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str, float]:
        """Validate model for one epoch."""
        key_mapping = self.config.train.key_mapping.validation

        if hasattr(key_mapping, "text_to_feat"):
            text_to_feat_key_mapping = key_mapping.text_to_feat
        else:
            text_to_feat_key_mapping = key_mapping

        criterion_names = self.criterion_names(self.config.criterion)
        mean_metrics = {
            criterion_name: MeanMetric(device=self.device) for criterion_name in criterion_names
        }
        n_batch = 0

        self.model.eval()

        if self.feat_to_wave is not None:
            self.feat_to_wave.eval()

            if self.transform_middle is not None and isinstance(self.transform_middle, nn.Module):
                self.transform_middle.eval()

        for named_data in self.loaders.validation:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=text_to_feat_key_mapping)
            named_target = self.map_to_named_target(named_data)
            output = self.model(**named_input)
            named_output = self.map_to_named_output(output, key_mapping=text_to_feat_key_mapping)

            if self.feat_to_wave is not None:
                (
                    named_transform_middle_output,
                    named_feat_to_wave_output,
                ) = self.feat_to_wave_forward(named_data, named_output, key_mapping=key_mapping)

                assert (
                    set(named_output.keys()) & set(named_transform_middle_output.keys()) == set()
                ), "named_output and named_transform_middle_output should be disjointed."
                assert (
                    set(named_output.keys()) & set(named_feat_to_wave_output.keys()) == set()
                ), "named_output and named_feat_to_wave_output should be disjointed."
                assert (
                    set(named_transform_middle_output.keys())
                    & set(named_feat_to_wave_output.keys())
                    == set()
                ), (
                    "named_transform_middle_output and named_feat_to_wave_output "
                    "should be disjointed."
                )

                # update named_output by outputs of feat-to-wave model
                named_output.update(named_transform_middle_output)
                named_output.update(named_feat_to_wave_output)

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

        if hasattr(inference_key_mapping, "text_to_feat"):
            text_to_feat_key_mapping = inference_key_mapping.text_to_feat
        else:
            text_to_feat_key_mapping = inference_key_mapping

        n_batch = 0

        self.model.eval()

        if self.feat_to_wave is not None:
            self.feat_to_wave.eval()

            if self.transform_middle is not None and isinstance(self.transform_middle, nn.Module):
                self.transform_middle.eval()

        for named_data in self.loaders.validation:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=text_to_feat_key_mapping)

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(output, key_mapping=text_to_feat_key_mapping)

            if self.feat_to_wave is not None:
                (
                    named_transform_middle_output,
                    named_feat_to_wave_output,
                ) = self.feat_to_wave_forward(
                    named_data,
                    named_output,
                    key_mapping=inference_key_mapping,
                    forward_method="inference",
                )

                assert (
                    set(named_output.keys()) & set(named_transform_middle_output.keys()) == set()
                ), "named_output and named_transform_middle_output should be disjointed."
                assert (
                    set(named_output.keys()) & set(named_feat_to_wave_output.keys()) == set()
                ), "named_output and named_feat_to_wave_output should be disjointed."
                assert (
                    set(named_transform_middle_output.keys())
                    & set(named_feat_to_wave_output.keys())
                    == set()
                ), (
                    "named_transform_middle_output and named_feat_to_wave_output "
                    "should be disjointed."
                )

                # update named_output by outputs of feat-to-wave model
                named_output.update(named_transform_middle_output)
                named_output.update(named_feat_to_wave_output)

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

    def feat_to_wave_forward(
        self,
        named_data: Dict[str, torch.Tensor],
        named_output: Dict[str, torch.Tensor],
        key_mapping: DictConfig = None,
        forward_method: Optional[str] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Perform forward pass of transform_middle and feat_to_wave.

        Args:
            named_data (dict): Dictionary data given by data loader.
            named_output (dict): Output of text-to-feat model.
            key_mapping (DictConfig): Mapping rule of outputs
                in transform_middle and feat_to_wave.
            forward_method (str, optional): If given, ``forward_method``
                is used. Otherwise, ``__call__`` (typically equivalent
                to ``forward``) is used.

        Returns:
            tuple: Tuple containing

                - dict: Output of transform_middle.
                - dict: Output of feat_to_wave.

        """
        if key_mapping is None:
            raise ValueError("Specify key_mapping.")

        if self.transform_middle is None:
            named_transform_middle_output = {}
        elif hasattr(key_mapping, "transform_middle") and key_mapping.transform_middle is not None:
            # transform middle
            transform_middle_key_mapping = key_mapping.transform_middle

            named_transform_middle_input = self.map_to_named_input(
                named_data,
                key_mapping=transform_middle_key_mapping,
                strict=False,
            )
            named_transform_middle_input_from_named_output = self.map_to_named_input(
                named_output,
                key_mapping=transform_middle_key_mapping,
                strict=False,
            )

            assert (
                set(named_transform_middle_input.keys())
                & set(named_transform_middle_input_from_named_output.keys())
                == set()
            ), (
                "named_transform_middle_input and named_transform_middle_input_from_named_output "
                "should be disjointed."
            )

            named_transform_middle_input.update(named_transform_middle_input_from_named_output)

            if forward_method is None:
                forward_fn = self.transform_middle
            else:
                if hasattr(self.transform_middle, forward_method):
                    unwrapped_transform_middle = unwrap(self.transform_middle)
                    forward_fn = getattr(unwrapped_transform_middle, forward_method)
                else:
                    forward_fn = self.transform_middle

            transform_middle_output = forward_fn(**named_transform_middle_input)
            named_transform_middle_output = self.map_to_named_output(
                transform_middle_output, key_mapping=transform_middle_key_mapping
            )
        else:
            named_transform_middle_output = {}

        # feat-to-wave
        if hasattr(key_mapping, "feat_to_wave") and key_mapping.feat_to_wave is not None:
            feat_to_wave_key_mapping = key_mapping.feat_to_wave

            named_feat_to_wave_input = self.map_to_named_input(
                named_data,
                key_mapping=feat_to_wave_key_mapping,
                strict=False,
            )
            named_feat_to_wave_input_from_named_output = self.map_to_named_input(
                named_output,
                key_mapping=feat_to_wave_key_mapping,
                strict=False,
            )
            named_feat_to_wave_input_from_named_transform_middle_output = self.map_to_named_input(
                named_transform_middle_output,
                key_mapping=feat_to_wave_key_mapping,
                strict=False,
            )

            assert (
                set(named_feat_to_wave_input.keys())
                & set(named_feat_to_wave_input_from_named_output.keys())
                == set()
            ), (
                "named_feat_to_wave_input and named_feat_to_wave_input_from_named_output "
                "should be disjointed."
            )
            assert (
                set(named_feat_to_wave_input.keys())
                & set(named_feat_to_wave_input_from_named_transform_middle_output.keys())
                == set()
            ), (
                "named_feat_to_wave_input and "
                "named_feat_to_wave_input_from_named_transform_middle_output "
                "should be disjointed."
            )
            assert (
                set(named_feat_to_wave_input_from_named_output.keys())
                & set(named_feat_to_wave_input_from_named_transform_middle_output.keys())
                == set()
            ), (
                "named_feat_to_wave_input_from_named_output and "
                "named_feat_to_wave_input_from_named_transform_middle_output "
                "should be disjointed."
            )

            named_feat_to_wave_input.update(named_feat_to_wave_input_from_named_output)
            named_feat_to_wave_input.update(
                named_feat_to_wave_input_from_named_transform_middle_output
            )

            if forward_method is None:
                forward_fn = self.feat_to_wave
            else:
                if hasattr(self.feat_to_wave, forward_method):
                    unwrapped_feat_to_wave = unwrap(self.feat_to_wave)
                    forward_fn = getattr(unwrapped_feat_to_wave, forward_method)
                else:
                    forward_fn = self.feat_to_wave

            feat_to_wave_output = forward_fn(**named_feat_to_wave_input)
            named_feat_to_wave_output = self.map_to_named_output(
                feat_to_wave_output, key_mapping=feat_to_wave_key_mapping
            )
        else:
            named_feat_to_wave_output = {}

        return named_transform_middle_output, named_feat_to_wave_output
