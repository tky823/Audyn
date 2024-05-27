import os
from typing import Optional, Union

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from .collator import Collator, default_collate_fn, rename_webdataset_keys
from .composer import AudioFeatureExtractionComposer, Composer, SequentialComposer
from .dataloader import (
    DistributedDataLoader,
    DistributedDynamicBatchDataLoader,
    DistributedSequentialBatchDataLoader,
    DynamicBatchDataLoader,
    SequentialBatchDataLoader,
)
from .dataset import SortableTorchObjectDataset, TorchObjectDataset, WebDatasetWrapper
from .postprocess import make_noise, slice_feautures, take_log_features

__all__ = [
    "TorchObjectDataset",
    "SortableTorchObjectDataset",
    "WebDatasetWrapper",
    "SequentialBatchDataLoader",
    "DistributedDataLoader",
    "DistributedSequentialBatchDataLoader",
    "DynamicBatchDataLoader",
    "DistributedDynamicBatchDataLoader",
    "Composer",
    "AudioFeatureExtractionComposer",
    "SequentialComposer",
    "Collator",
    "slice_feautures",
    "take_log_features",
    "make_noise",
    "default_collate_fn",
    "rename_webdataset_keys",
]


class BaseDataLoaders:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> None:
        self.train = train_loader
        self.validation = validation_loader


def select_accelerator(config_or_accelerator: Union[DictConfig, Optional[str]]) -> str:
    """Select accelerator from system config or accelerator string.

    Args:
        config_or_accelerator (DictConfig or str): If ``DictConfig`` is given,
            check ``accelerator`` attribute. If ``None`` is given,
            select ``cuda`` or ``cpu`` automatically dependeing on environment.

    Returns:
        str: String representing accelerator.

    """
    if isinstance(config_or_accelerator, DictConfig):
        system_config = config_or_accelerator
        accelerator = system_config.accelerator
    else:
        accelerator = config_or_accelerator

    if accelerator is None:
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    return accelerator


def select_device(accelerator: Optional[str], is_distributed: bool = False) -> Union[str, int]:
    """Select device by accelerator."""
    accelerator = select_accelerator(accelerator)

    if accelerator in ["cuda", "gpu"] and is_distributed:
        device = int(os.environ["LOCAL_RANK"])
    elif accelerator in ["cpu", "cuda", "mps"]:
        device = accelerator
    elif accelerator == "gpu":
        device = "cuda"
    else:
        raise ValueError(f"Accelerator {accelerator} is not supported.")

    return device
