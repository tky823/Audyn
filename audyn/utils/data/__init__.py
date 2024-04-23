import os
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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


def default_collate_fn(
    list_batch: List[Dict[str, torch.Tensor]], keys: Optional[Iterable[str]] = None
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.
    """
    if keys is None:
        for data in list_batch:
            if keys is None:
                keys = set(data.keys())
            else:
                assert set(keys) == set(data.keys())

    dict_batch = {key: [] for key in keys}
    tensor_keys = set()
    pad_keys = set()

    for data in list_batch:
        for key in keys:
            if isinstance(data[key], torch.Tensor):
                tensor_keys.add(key)

                if data[key].dim() > 0:
                    pad_keys.add(key)
                    data[key] = torch.swapaxes(data[key], 0, -1)

            dict_batch[key].append(data[key])

    for key in keys:
        if key in pad_keys:
            dict_batch[key] = nn.utils.rnn.pad_sequence(dict_batch[key], batch_first=True)
            dict_batch[key] = torch.swapaxes(dict_batch[key], 1, -1)
        elif key in tensor_keys:
            dict_batch[key] = torch.stack(dict_batch[key], dim=0)

    dict_batch = rename_webdataset_keys(dict_batch)

    return dict_batch


def rename_webdataset_keys(dict_batch: Dict[str, Any]) -> Dict[str, Any]:
    keys = list(dict_batch.keys())

    for key in keys:
        webdataset_key = _rename_webdataset_key_if_possible(key)

        if webdataset_key != key:
            dict_batch[webdataset_key] = dict_batch.pop(key)

    return dict_batch


def _rename_webdataset_key_if_possible(key: str) -> str:
    if "." in key:
        if len(key.split(".")) > 2:
            raise NotImplementedError("Multiple dots in a key is not supported.")

        # remove extension
        key, _ = key.split(".")

    return key
