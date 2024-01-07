import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data.dataloader import DataLoader

from ...utils.logging import get_logger
from .base import BaseGenerator, BaseTrainer

__all__ = [
    "TextToWaveTrainer",
    "CascadeTextToWaveGenerator",
]


class TextToWaveTrainer(BaseTrainer):
    """Trainer for text-to-wave model."""


class CascadeTextToWaveGenerator(BaseGenerator):
    def __init__(
        self,
        loader: DataLoader,
        model: nn.Module,
        config: DictConfig = None,
    ) -> None:
        super().__init__(loader, model, config=config)

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        self.exp_dir = config.test.output.exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)

        self.inference_dir = config.test.output.inference_dir
        os.makedirs(self.inference_dir, exist_ok=True)

        # Set loggder
        self.logger = get_logger(self.__class__.__name__, is_distributed=self.is_distributed)

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        text_to_feat_checkpoint = config.test.checkpoint.text_to_feat
        feat_to_wave_checkpoint = config.test.checkpoint.feat_to_wave

        self.logger.info(f"Load weights of text to feat model: {text_to_feat_checkpoint}.")
        self.logger.info(f"Load weights of feat to wave model: {feat_to_wave_checkpoint}.")
        self.load_checkpoint(text_to_feat_checkpoint, feat_to_wave_checkpoint)

        self.remove_weight_norm_if_necessary()

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        for named_batch in self.loader:
            named_batch = self.move_data_to_device(named_batch, self.device)
            named_input = self.map_to_named_input(
                named_batch, key_mapping=self.config.test.key_mapping.inference
            )
            named_identifier = self.map_to_named_identifier(
                named_batch, key_mapping=self.config.test.key_mapping.inference
            )

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(
                output, key_mapping=self.config.test.key_mapping.inference
            )

            self.save_inference_audio_if_necessary(
                named_output,
                named_batch,
                named_identifier,
                config=self.config.test.output,
            )
            self.save_inference_spectrogram_if_necessary(
                named_output,
                named_batch,
                named_identifier,
                config=self.config.test.output,
            )

    def load_checkpoint(self, text_to_feat_path, feat_to_wave_path: str) -> None:
        text_to_feat_state_dict = torch.load(text_to_feat_path, map_location=self.device)
        feat_to_wave_state_dict = torch.load(feat_to_wave_path, map_location=self.device)
        unwrapped_text_to_feat = self.unwrapped_model.text_to_feat
        unwrapped_feat_to_wave = self.unwrapped_model.feat_to_wave

        if (
            "generator" in text_to_feat_state_dict["model"]
            and "discriminator" in text_to_feat_state_dict["model"]
        ):
            unwrapped_text_to_feat.load_state_dict(text_to_feat_state_dict["model"]["generator"])
        else:
            unwrapped_text_to_feat.load_state_dict(text_to_feat_state_dict["model"])

        if (
            "generator" in feat_to_wave_state_dict["model"]
            and "discriminator" in feat_to_wave_state_dict["model"]
        ):
            unwrapped_feat_to_wave.load_state_dict(feat_to_wave_state_dict["model"]["generator"])
        else:
            unwrapped_feat_to_wave.load_state_dict(feat_to_wave_state_dict["model"])

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
