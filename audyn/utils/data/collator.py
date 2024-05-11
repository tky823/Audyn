from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from .dataset import Composer

__all__ = [
    "Collator",
    "default_collate_fn",
    "rename_webdataset_keys",
]


class Collator:
    """Base class of collator."""

    def __init__(self, composer: Optional[Composer] = None) -> None:
        self.composer = composer

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        composer = self.composer

        if composer is None:
            list_batch = batch
        else:
            list_batch = []

            for sample in composer(batch):
                list_batch.append(sample)

        dict_batch = default_collate_fn(list_batch)

        return dict_batch


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


def rename_webdataset_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """Rename keys of WebDataset.

    Args:
        data (dict): Dictionary-like batch or sample.

    Returns:
        dict: Dictionary-like batch or sample with renamed keys.

    Examples:

        >>> import torch
        >>> from audyn.utils.data import rename_webdataset_keys
        >>> data = {"audio.m4a": torch.tensor((2, 16000))}
        >>> data.keys()
        dict_keys(['audio.m4a'])
        >>> data = rename_webdataset_keys(data)
        >>> data.keys()
        dict_keys(['audio'])

    """
    keys = list(data.keys())

    for key in keys:
        webdataset_key = _rename_webdataset_key_if_possible(key)

        if webdataset_key != key:
            data[webdataset_key] = data.pop(key)

    return data


def _rename_webdataset_key_if_possible(key: str) -> str:
    if "." in key:
        if len(key.split(".")) > 2:
            raise NotImplementedError("Multiple dots in a key is not supported.")

        # remove extension
        key, _ = key.split(".")

    return key
