import os

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from tqdm import tqdm  # noqa: F811

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False

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

        feature_dir = self.config.preprocess.feature_dir
        filename_template = train_config.output.filename

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

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])
