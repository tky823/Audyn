import os
from typing import Dict, List, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ..logging import get_logger
from .base import BaseDriver


class CrossModalFeatureSaver(BaseDriver):
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
        key_mapping = self.config.test.key_mapping.inference

        assert self.loader.batch_size == 1, "batch_size should be 1 for correct aggregation."

        self.model.eval()

        for named_data in self.loader:
            named_data = self.move_data_to_device(named_data, self.device)
            named_input = self.map_to_named_input(named_data, key_mapping=key_mapping)
            named_identifier = self.map_to_named_identifier(
                named_data, key_mapping=self.config.test.key_mapping.inference
            )

            if hasattr(self.unwrapped_model, "inference"):
                output = self.unwrapped_model.inference(**named_input)
            else:
                output = self.unwrapped_model(**named_input)

            named_output = self.map_to_named_output(output, key_mapping=key_mapping)

            self.save_named_feature_if_necessary(named_output, named_data, named_identifier)

    def load_checkpoint(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)

        self.unwrapped_model.load_state_dict(state_dict["model"])

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

    def save_named_feature_if_necessary(
        self,
        named_output: Dict[str, torch.Tensor],
        named_reference: Dict[str, torch.Tensor],
        named_identifier: Dict[str, List[str]],
        key_mapping: DictConfig = None,
        transforms: DictConfig = None,
    ) -> None:
        identifier_keys = named_identifier.keys()

        output_features = []
        reference_features = []

        if hasattr(key_mapping, "output") and key_mapping.output is not None:
            if named_output is None:
                raise ValueError("named_output is not specified.")

            for key, filename in key_mapping.output.items():
                for idx, feature in enumerate(named_output[key]):
                    if transforms is not None and transforms.output is not None:
                        if key in transforms.output.keys():
                            transform = hydra.utils.instantiate(transforms.output[key])
                            feature = transform(feature)

                    if len(output_features) > idx:
                        output_features.append({key: feature})
                    else:
                        output_features[idx][key] = feature

        if hasattr(key_mapping, "reference") and key_mapping.reference is not None:
            if named_reference is None:
                raise ValueError("named_reference is not specified.")

            for key, filename in key_mapping.reference.items():
                for feature in named_reference[key]:
                    if transforms is not None and transforms.reference is not None:
                        if key in transforms.reference.keys():
                            transform = hydra.utils.instantiate(transforms.reference[key])
                            feature = transform(feature)

                    if len(reference_features) > idx:
                        reference_features.append({key: feature})
                    else:
                        reference_features[idx][key] = feature
