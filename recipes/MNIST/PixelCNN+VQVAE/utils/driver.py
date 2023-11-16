import os
from typing import Dict, List

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models.cascade import PixelCNNVQVAE

try:
    from tqdm import tqdm  # noqa: F811

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False

from audyn.utils.driver import BaseGenerator
from audyn.utils.driver._decorator import run_only_master_rank
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

        list_path = self.config.preprocess.list_path
        feature_dir = self.config.preprocess.feature_dir
        filename_template = train_config.output.filename
        filenames = []

        pbar = enumerate(self.loader)

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(pbar)

        for batch_idx, named_batch in pbar:
            for data_key in named_batch.keys():
                batch_size = len(named_batch[data_key])

                assert batch_size == 1

            named_batch = self.move_data_to_device(named_batch, self.device)
            named_input = self.map_to_named_input(
                named_batch, key_mapping=train_config.key_mapping
            )

            with autocast(enabled=self.enable_amp):
                output = self.model(**named_input)

            named_output = self.map_to_named_output(
                output,
                key_mapping=train_config.key_mapping,
            )

            for sample_idx in range(batch_size):
                filename = filename_template.format(number=batch_idx + 1)
                data = {"filename": filename}

                for save_key in train_config.key_mapping.save.keys():
                    output_key = train_config.key_mapping.save[save_key]
                    data[save_key] = named_output[output_key][sample_idx]

                path = os.path.join(feature_dir, f"{filename}.pth")
                save_dir = os.path.dirname(path)
                os.makedirs(save_dir, exist_ok=True)

                torch.save(data, path)

                filenames.append(filename)

        list_dir = os.path.dirname(list_path)
        os.makedirs(list_dir, exist_ok=True)

        with open(list_path, mode="w") as f:
            for filename in filenames:
                f.write(filename + "\n")

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])


class Generator(BaseGenerator):
    model: PixelCNNVQVAE

    def __init__(
        self,
        loader: DataLoader,
        model: PixelCNNVQVAE,
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

        self.logger.info(f"Load weights of PixelCNN: {checkpoint.pixelcnn}.")
        self.logger.info(f"Load weights of VQ-VAE: {checkpoint.vqvae}.")
        self.load_checkpoint(checkpoint.pixelcnn, checkpoint.vqvae)

        self.remove_weight_norm_if_necessary()

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        pbar = self.loader

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(pbar)

        for named_batch in pbar:
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

            if hasattr(self.config.test.output, "figure"):
                figure_config = self.config.test.output.figure

                if figure_config is not None:
                    if hasattr(figure_config.key_mapping, "inference"):
                        key_mapping = figure_config.key_mapping.inference
                    elif hasattr(figure_config.key_mapping, "test"):
                        key_mapping = figure_config.key_mapping.test
                    else:
                        key_mapping = figure_config.key_mapping

                    if hasattr(figure_config.key_mapping, "inference"):
                        transforms = figure_config.transforms.inference
                    elif hasattr(figure_config.key_mapping, "test"):
                        transforms = figure_config.transforms.test
                    else:
                        transforms = figure_config.transforms

                    self.save_figure_if_necessary(
                        named_output,
                        named_batch,
                        named_identifier,
                        key_mapping=key_mapping,
                        transforms=transforms,
                    )

    def load_checkpoint(self, pixelcnn_path: str, vqvae_path: str) -> None:
        # load weights of PixelCNN
        state_dict = torch.load(pixelcnn_path, map_location=self.device)
        self.unwrapped_model.pixelcnn.load_state_dict(state_dict["model"])

        # load weights of PixelCNN
        state_dict = torch.load(vqvae_path, map_location=self.device)
        self.unwrapped_model.vqvae.load_state_dict(state_dict["model"])

    @run_only_master_rank()
    def save_figure_if_necessary(
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
                for idx, image in enumerate(named_output[key]):
                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = hydra.utils.instantiate(transforms.output[key])
                            image = transform(image)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_figure(path, image)

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, filename in key_mapping.reference.items():
                for image in named_reference[key]:
                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = hydra.utils.instantiate(transforms.reference[key])
                            image = transform(image)

                    identifier_mapping = {
                        identifier_key: named_identifier[identifier_key][idx]
                        for identifier_key in identifier_keys
                    }
                    path = os.path.join(self.inference_dir, filename)
                    path = path.format(**identifier_mapping)
                    self.save_figure(path, image)

    def save_figure(
        self,
        path: str,
        image: torch.Tensor,
    ) -> None:
        """Save figure using matplotlib."""
        assert image.dim() in [
            2,
            3,
        ], f"image is expected to be 2 or 3D tesor, but given as {image.dim()}D tensor."

        image = image.detach().cpu()

        if image.dim() == 3:
            image = image.squeeze(dim=0)

        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap="gray")
        fig.savefig(path, bbox_inches="tight")
        plt.close()
