from typing import Any, Dict, Optional, Union

import torch

__all__ = ["slice_feautures"]


def slice_feautures(
    batch: Dict[str, Any],
    slice_length: int,
    key_mapping: Optional[Dict[str, str]] = None,
    hop_lengths: Optional[Dict[str, int]] = None,
    length_dims: Optional[Union[int, Dict[str, int]]] = None,
    random_slice: bool = False,
) -> Dict[str, Any]:
    """Add sliced features from given batch.

    Args:
        batch (dict): Dict-type batch.
        slice_length (int): Length of sliced waveform.
        key_mapping (dict, optional): Mapping of keys to sliced feature.
        hop_lengths (dict, optional): Unit hop lengths of features.
        length_dims (int or dict, optional): Dimension to get length of features.
        random_slice (bool): If ``True``, slice section is selected at random.

    Returns:
        dict: Dict-type batch including sliced features.

    """
    if key_mapping is None:
        key_mapping = {}

    # hop length
    if hop_lengths is None:
        hop_lengths = {}

    for key in key_mapping.keys():
        if key not in hop_lengths:
            hop_lengths[key] = 1

    # find low resolution (= largest hop length) feature
    low_resolution_key = None

    for key in key_mapping.keys():
        if low_resolution_key is None:
            low_resolution_key = key
        elif hop_lengths[key] > hop_lengths[low_resolution_key]:
            low_resolution_key = key

    if length_dims is None:
        length_dims = {key: -1 for key in key_mapping.keys()}
    else:
        if isinstance(length_dims, int):
            length_dims = {key: length_dims for key in key_mapping.keys()}

    # obtain batch size
    if len(key_mapping) > 0:
        batch_size_keys = sorted(list(key_mapping.keys()))
    else:
        batch_size_keys = sorted(list(batch.keys()))

    batch_size_key = batch_size_keys[0]
    batch_size = len(batch[batch_size_key])

    for slice_key in key_mapping.values():
        batch[slice_key] = []

    for sample_idx in range(batch_size):
        key = low_resolution_key
        feature = batch[key][sample_idx]
        length = feature.size(length_dims[key])
        hop_length = hop_lengths[key]

        if random_slice:
            start_idx = torch.randint(0, length - slice_length // hop_length, (), dtype=torch.long)
        else:
            start_idx = length // 2 - (slice_length // hop_length) // 2

        end_idx = start_idx + slice_length // hop_length
        slice_key = key_mapping[key]

        _, sliced_feature, _ = torch.split(
            feature,
            [start_idx, end_idx - start_idx, length - end_idx],
            dim=-1,
        )

        batch[slice_key].append(sliced_feature)

        for key in key_mapping.keys():
            if key == slice_key:
                continue

            feature = batch[key][sample_idx]
            length = feature.size(length_dims[key])
            hop_length = hop_lengths[key]

            end_idx = start_idx + slice_length // hop_length
            slice_key = key_mapping[key]

            _, sliced_feature, _ = torch.split(
                feature,
                [start_idx, end_idx - start_idx, length - end_idx],
                dim=-1,
            )

            batch[slice_key].append(sliced_feature)

    for slice_key in key_mapping.values():
        batch[slice_key] = torch.stack(batch[slice_key], dim=0)

    return batch
