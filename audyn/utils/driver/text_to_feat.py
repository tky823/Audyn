from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ...utils.hydra.utils import instantiate_model
from ...utils.model import set_device
from ..data import BaseDataLoaders
from .base import BaseTrainer


class TextToFeatTrainer(BaseTrainer):
    """Trainer for text-to-feat model."""

    def __init__(
        self,
        loaders: BaseDataLoaders,
        model: nn.Module,
        optimizer: Optimizer,
        lr_scheduler: Optional[_LRScheduler] = None,
        criterion: Dict[str, nn.Module] = None,
        config: DictConfig = None,
    ) -> None:
        super().__init__(
            loaders,
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
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

            if hasattr(pretrained_config, "path") and pretrained_config.path is not None:
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
                )
                feat_to_wave.load_state_dict(state_dict["model"])
            else:
                # TODO support generator of GAN
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
                    )

        self.feat_to_wave = feat_to_wave
        self.transform_middle = transform_middle

    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str, float]:
        """Validate model for one epoch."""
        record_config = self.config.train.record
        criterion_names = {
            key
            for key in self.config.criterion.keys()
            if not key.startswith("_") and not key.endswith("_")
        }
        validation_loss = {criterion_name: 0 for criterion_name in criterion_names}
        n_batch = 0

        self.model.eval()

        if self.feat_to_wave is not None:
            self.feat_to_wave.eval()

            if self.transform_middle is not None and isinstance(self.transform_middle, nn.Module):
                self.transform_middle.eval()

        for named_data in self.loaders.validation:
            key_mapping = self.config.train.key_mapping.validation

            if hasattr(key_mapping, "text_to_feat"):
                text_to_feat_key_mapping = key_mapping.text_to_feat
            else:
                text_to_feat_key_mapping = key_mapping

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
                validation_loss[criterion_name] = (
                    validation_loss[criterion_name] + loss[criterion_name].item()
                )

            if hasattr(record_config, "spectrogram") and n_batch < 1:
                spectrogram_config = record_config.spectrogram.epoch
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
                        named_data,
                        sample_size=spectrogram_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

            if hasattr(record_config, "waveform") and n_batch < 1:
                waveform_config = record_config.waveform.epoch
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
                        named_data,
                        sample_size=waveform_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

            if hasattr(record_config, "audio") and n_batch < 1:
                audio_config = record_config.audio.epoch
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
                        named_data,
                        sample_size=audio_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                        sample_rate=audio_config.sample_rate,
                    )

            if hasattr(record_config, "image") and n_batch < 1:
                image_config = record_config.image.epoch
                global_step = self.epoch_idx + 1

                if hasattr(image_config.key_mapping, "validation"):
                    key_mapping = image_config.key_mapping.validation
                else:
                    key_mapping = image_config.key_mapping

                if hasattr(image_config.transforms, "validation"):
                    transforms = image_config.transforms.validation
                else:
                    transforms = image_config.transforms

                if image_config is not None and global_step % image_config.every == 0:
                    self.write_image_if_necessary(
                        named_output,
                        named_data,
                        sample_size=image_config.sample_size,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        global_step=global_step,
                    )

            n_batch += 1

        for criterion_name in criterion_names:
            validation_loss[criterion_name] = validation_loss[criterion_name] / n_batch

        return validation_loss

    def feat_to_wave_forward(
        self, named_data, named_output, key_mapping: DictConfig = None
    ) -> None:
        """Perform forward pass of transform_middle and feat_to_wave."""
        if key_mapping is None:
            raise ValueError("Specify key_mapping.")

        if self.transform_middle is None:
            named_transform_middle_output = {}
        else:
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

            transform_middle_output = self.transform_middle(**named_transform_middle_input)
            named_transform_middle_output = self.map_to_named_output(
                transform_middle_output, key_mapping=transform_middle_key_mapping
            )

        # feat-to-wave
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

        feat_to_wave_output = self.feat_to_wave(**named_feat_to_wave_input)
        named_feat_to_wave_output = self.map_to_named_output(
            feat_to_wave_output, key_mapping=feat_to_wave_key_mapping
        )

        return named_transform_middle_output, named_feat_to_wave_output
