# based on https://github.com/tky823/Audyn/blob/02ead2dc37f377dac0a60ae9adb1c71f019945d2/recipes/DCASE2023FoleySoundSynthesis/Baseline/utils/driver.py  # noqa: E501
import os

import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models.clap import CLAP

try:
    from tqdm import tqdm  # noqa: F811

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


from audyn.utils.driver.base import BaseDriver
from audyn.utils.logging import get_logger


class EmbeddingSaver(BaseDriver):
    def __init__(self, loader: DataLoader, model: CLAP, config: DictConfig = None) -> None:
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

        checkpoint = config.test.checkpoint
        self.logger.info(f"Load weights of model: {checkpoint}.")
        self.load_checkpoint(checkpoint)

    @torch.no_grad()
    def run(self) -> None:
        self.model.eval()

        test_key_mapping = self.config.test.key_mapping
        feature_dir = self.config.preprocess.feature_dir

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(self.loader)
        else:
            pbar = self.loader

        for named_data in pbar:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=test_key_mapping)

            with autocast(enabled=self.enable_amp):
                output = self.model(**named_input)

            named_output = self.map_to_named_output(
                output,
                key_mapping=test_key_mapping,
            )

            data = {}

            for save_key in test_key_mapping.save.input.keys():
                input_key = test_key_mapping.save.input[save_key]
                data[save_key] = named_data[input_key]

            for save_key in test_key_mapping.save.output.keys():
                output_key = test_key_mapping.save.output[save_key]
                data[save_key] = named_output[output_key]

            assert len(data["identifier"]) == 1

            identifiler = data["identifier"][0]
            path = os.path.join(feature_dir, f"{identifiler}.pth")
            save_dir = os.path.dirname(path)
            os.makedirs(save_dir, exist_ok=True)

            torch.save(data, path)

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])
