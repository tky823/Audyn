# based on https://github.com/tky823/Audyn/blob/02ead2dc37f377dac0a60ae9adb1c71f019945d2/recipes/DCASE2023FoleySoundSynthesis/Baseline/utils/driver.py  # noqa: E501
import copy
import os
from typing import Dict, List, Optional, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.models.clap import CLAP

from audyn.amp import autocast, get_autocast_device_type
from audyn.metrics import MultiMetrics, StatefulMetric
from audyn.utils.driver.base import BaseDriver
from audyn.utils.logging import get_logger

try:
    from tqdm import tqdm  # noqa: F811

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


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

            device_type = get_autocast_device_type()

            with autocast(device_type, enabled=self.enable_amp, dtype=self.amp_dtype):
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
        state_dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.unwrapped_model.load_state_dict(state_dict["model"])


class RetrievalTester(BaseDriver):
    def __init__(
        self,
        loader: DataLoader,
        metrics: Union[MultiMetrics, Dict[str, StatefulMetric]],
        config: DictConfig = None,
    ) -> None:
        self.loader = loader
        self.metrics = metrics

        self.config = config

        self._reset(config)

    def metric_names(self, config: Optional[DictConfig] = None) -> List[str]:
        if config is None:
            config = self.config.test.metrics

        names = [key for key in config.keys() if not key.startswith("_") and not key.endswith("_")]
        names = sorted(names)

        return names

    def _reset(self, config: DictConfig) -> None:
        self.set_system(config=config.system)

        # Set loggder
        self.logger = get_logger(self.__class__.__name__, is_distributed=self.is_distributed)

        assert self.device == "cpu"

    @torch.no_grad()
    def run(self) -> None:
        test_config: DictConfig = self.config.test
        test_key_mapping = test_config.key_mapping
        target_keys = test_key_mapping.target
        query_keys = test_key_mapping.query
        metric_names = self.metric_names(test_config.metrics)

        self.metrics.reset()

        named_target = {key: [] for key in target_keys}

        for named_data in self.loader:
            for key in target_keys:
                assert isinstance(named_data[key], torch.Tensor)

                named_target[key].append(named_data[key])

        for key in target_keys:
            named_target[key] = torch.cat(named_target[key], dim=0)

        named_target = self.move_data_to_device(named_target, self.device)
        named_input = {}

        for name in metric_names:
            named_input[name] = {}
            metric_config = getattr(test_config.metrics, name)

            # NOTE: metric_config.key_mapping.target here represents keys derived from named_data,
            #       while named_target represents target to retrieve.
            for metric_key, data_key in metric_config.key_mapping.target.items():
                if data_key in named_target.keys():
                    named_input[name][metric_key] = named_target[data_key]

        if IS_TQDM_AVAILABLE:
            pbar = tqdm(self.loader)
        else:
            pbar = self.loader

        next_sample_idx = 0

        for named_data in pbar:
            named_data = self.move_data_to_device(named_data, self.device)

            batch_size = None
            named_query = {}

            for key in query_keys:
                assert isinstance(named_data[key], torch.Tensor)

                named_query[key] = named_data[key]

                if batch_size is None:
                    batch_size = named_query[key].size(0)
                else:
                    assert named_query[key].size(0) == batch_size

            if batch_size is None:
                raise ValueError("Batch size cannot be determined.")

            for name in metric_names:
                metric_config = getattr(test_config.metrics, name)
                metric_query_keys = []
                named_metric_input = copy.deepcopy(named_input[name])

                for metric_key, data_key in metric_config.key_mapping.target.items():
                    if data_key in named_query.keys():
                        metric_query_keys.append(metric_key)

                        assert metric_key not in named_metric_input

                        named_metric_input[metric_key] = named_query[data_key]

                # TODO: remove hardcode
                named_metric_input["index"] = torch.tensor(next_sample_idx)
                named_metric_input["index"] = named_metric_input["index"].expand(batch_size)

                self.metrics[name].update(**named_metric_input)

            # NOTE: Batch size is assumed to be 1.
            next_sample_idx += 1

        s = ""

        for metric_name in metric_names:
            metric = self.metrics[metric_name]
            loss = metric.compute().item()

            s += f"{metric_name}: {loss}, "

        self.logger.info(s[:-2])
