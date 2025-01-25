import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models.cascade import BaselineModel

try:
    from tqdm import tqdm  # noqa: F811

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


from audyn.amp import autocast, get_autocast_device_type
from audyn.utils.driver import BaseGenerator
from audyn.utils.driver.base import BaseDriver
from audyn.utils.logging import get_logger


class PriorSaver(BaseDriver):
    def __init__(self, loader: DataLoader, model: nn.Module, config: DictConfig) -> None:
        self.loader = loader
        self.model = model

        self.config = config

        self._reset(config)

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        # Set loggder
        self.logger = get_logger(self.__class__.__name__, is_distributed=self.is_distributed)

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        checkpoint = config.train.checkpoint
        self.logger.info(f"Load weights of model: {checkpoint}.")
        self.load_checkpoint(checkpoint)

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        train_config = self.config.train
        data_config = self.config.data

        feature_dir = self.config.preprocess.feature_dir

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(self.loader)
        else:
            pbar = self.loader

        for named_batch in pbar:
            for data_key in named_batch.keys():
                batch_size = len(named_batch[data_key])

                assert batch_size == 1

            named_batch = self.move_data_to_device(named_batch, self.device)
            named_input = self.map_to_named_input(
                named_batch, key_mapping=train_config.key_mapping
            )

            device_type = get_autocast_device_type()

            with autocast(device_type, enabled=self.enable_amp, dtype=self.amp_dtype):
                output = self.model(**named_input)

            named_output = self.map_to_named_output(
                output,
                key_mapping=train_config.key_mapping,
            )

            for sample_idx in range(batch_size):
                data = {}

                for save_key in train_config.key_mapping.save.input.keys():
                    input_key = train_config.key_mapping.save.input[save_key]
                    data[save_key] = named_batch[input_key][sample_idx]

                for save_key in train_config.key_mapping.save.output.keys():
                    output_key = train_config.key_mapping.save.output[save_key]
                    data[save_key] = named_output[output_key][sample_idx]

                    if save_key == "indices":
                        # validate shape of latent feature
                        assert data[save_key].size() == data_config.codebook.shape

                identifiler = data["identifier"]
                path = os.path.join(feature_dir, f"{identifiler}.pth")
                save_dir = os.path.dirname(path)
                os.makedirs(save_dir, exist_ok=True)

                torch.save(data, path)

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])


class Generator(BaseGenerator):
    model: BaselineModel

    def __init__(
        self,
        loader: DataLoader,
        model: BaselineModel,
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
        self.logger = get_logger(
            self.__class__.__name__,
            is_distributed=self.is_distributed,
        )

        # Display config and model architecture after logger instantiation
        self.logger.info(OmegaConf.to_yaml(self.config))
        self.display_model(display_num_parameters=True)

        checkpoint = config.test.checkpoint

        self.logger.info(f"Load weights of PixelSNAIL: {checkpoint.pixelsnail}.")
        self.logger.info(f"Load weights of VQVAE: {checkpoint.vqvae}.")
        self.logger.info(f"Load weights of HiFi-GAN: {checkpoint.hifigan}.")
        self.load_checkpoint(checkpoint.pixelsnail, checkpoint.vqvae, checkpoint.hifigan)

        self.remove_weight_norm_if_necessary()

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        pbar = self.loader

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(pbar)

        for named_data in pbar:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(
                named_data, key_mapping=self.config.test.key_mapping.inference
            )
            named_identifier = self.map_to_named_identifier(
                named_data, key_mapping=self.config.test.key_mapping.inference
            )

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(
                output, key_mapping=self.config.test.key_mapping.inference
            )

            if hasattr(self.config.test.output, "audio"):
                audio_config = self.config.test.output.audio

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
                        named_data,
                        named_identifier,
                        key_mapping=key_mapping,
                        transforms=transforms,
                        sample_rate=audio_config.sample_rate,
                    )

    def load_checkpoint(self, pixelsnail_path: str, vqvae_path: str, hifigan_path: str) -> None:
        """Load checkpoints of PixelSNAIL, VQVAE, and generator of HiFiGAN to baseline model.

        Args:
            pixelsnail_path (str): Path to pretrained PixelSNAIL.
            vqvae_path (str): Path to pretrained VQVAE.
            hifigan_path (str): Path to pretrained HiFi-GAN.

        """
        self.unwrapped_model: BaselineModel

        # load weights of PixelSNAIL
        state_dict = torch.load(pixelsnail_path, map_location=self.device)
        self.unwrapped_model.pixelsnail.load_state_dict(state_dict["model"])

        # load weights of VQVAE
        state_dict = torch.load(vqvae_path, map_location=self.device)
        self.unwrapped_model.vqvae.load_state_dict(state_dict["model"])

        # load weights of HiFi-GAN
        state_dict = torch.load(hifigan_path, map_location=self.device)
        self.unwrapped_model.hifigan_generator.load_state_dict(state_dict["model"]["generator"])
