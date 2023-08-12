import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataloader import (
    DistributedDynamicBatchDataLoader,
    DistributedSequentialBatchDataLoader,
    DynamicBatchDataLoader,
    SequentialBatchDataLoader,
)
from .dataset import SortableTorchObjectDataset, TorchObjectDataset
from .postprocess import slice_feautures

__all__ = [
    "TorchObjectDataset",
    "SortableTorchObjectDataset",
    "SequentialBatchDataLoader",
    "DistributedSequentialBatchDataLoader",
    "DynamicBatchDataLoader",
    "DistributedDynamicBatchDataLoader",
    "slice_feautures",
    "default_collate_fn",
]


class BaseDataLoaders:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> None:
        self.train = train_loader
        self.validation = validation_loader


def select_device(accelerator: str, is_distributed: bool = False) -> str:
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

    return dict_batch
