import math
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn.functional as F

__all__ = ["slice_feautures", "take_log_features", "make_noise"]


def slice_feautures(
    batch: Dict[str, Any],
    slice_length: int,
    key_mapping: Optional[Dict[str, str]] = None,
    hop_lengths: Optional[Dict[str, int]] = None,
    length_mapping: Optional[Dict[str, str]] = None,
    length_dims: Optional[Union[int, Dict[str, int]]] = None,
    random_slice: bool = False,
    pad_values: Optional[Dict[str, Any]] = None,
    inplace: bool = True,
) -> Dict[str, Any]:
    """Add sliced features from given batch.

    Args:
        batch (dict): Dict-type batch.
        slice_length (int): Length of sliced waveform.
        key_mapping (dict, optional): Mapping of keys to sliced feature.
        hop_lengths (dict, optional): Unit hop lengths of features.
        length_mapping  (dict, optional): Length mapping of features.
        length_dims (int or dict, optional): Dimension to get length of features.
        random_slice (bool): If ``True``, slice section is selected at random.
        pad_values (dict, optional): If given, shorter features are padded by these values.
        inplace (bool): If ``True``, sliced features are saved in given batch.
            Otherwise, new dictionary is prepared to avoid inplace operation.

    Returns:
        dict: Dict-type batch including sliced features.

    """

    def _compute_dim_without_batch(full_dim: int) -> int:
        if full_dim < 0:
            dim = full_dim
        elif length_dim > 0:
            dim = full_dim - 1
        else:
            raise ValueError("0 is batch dimension.")

        return dim

    if inplace:
        output_batch = batch
    else:
        output_batch = {key: value for key, value in batch.items()}

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

    if length_mapping is None:
        _length_mapping = {key: None for key in key_mapping.keys()}
    else:
        _length_mapping = {}

        for key in key_mapping.keys():
            if key in length_mapping.keys():
                _length_mapping[key] = length_mapping[key]
            else:
                _length_mapping[key] = None

    if length_dims is None:
        _length_dims = {key: -1 for key in key_mapping.keys()}
    else:
        if isinstance(length_dims, int):
            _length_dims = {key: length_dims for key in key_mapping.keys()}
        else:
            _length_dims = {}

            for key in key_mapping.keys():
                if key in _length_mapping.keys():
                    _length_dims[key] = length_dims[key]
                else:
                    _length_dims[key] = -1

    # obtain batch size
    if len(key_mapping) > 0:
        batch_size_keys = sorted(list(key_mapping.keys()))
    else:
        batch_size_keys = sorted(list(output_batch.keys()))

    batch_size_key = batch_size_keys[0]
    batch_size = len(output_batch[batch_size_key])

    for slice_key in key_mapping.values():
        output_batch[slice_key] = []

    for sample_idx in range(batch_size):
        key = low_resolution_key
        feature = output_batch[key][sample_idx]
        length_key = _length_mapping[key]
        length_dim = _length_dims[key]
        hop_length = hop_lengths[key]
        sliced_feature_length = math.ceil(slice_length / hop_length)
        _length_dim = _compute_dim_without_batch(length_dim)

        if _length_dim < 0:
            # set _length_dim to be non-negative
            _length_dim = feature.dim() + _length_dim

        length = _compute_length(
            output_batch,
            key,
            sample_idx,
            length_key=length_key,
            length_dim=_length_dim,
        )

        if length < sliced_feature_length:
            if pad_values is None or pad_values.get(key) is None:
                raise ValueError(
                    f"Input length ({length}) is shorter "
                    f"than slice length ({sliced_feature_length}) "
                    f"for {key} key."
                )
            else:
                # remove exising padding
                padding = feature.size(_length_dim) - length
                feature, _ = torch.split(
                    feature, [length, feature.size(_length_dim) - length], dim=_length_dim
                )
                # append padding
                padding = sliced_feature_length - length
                pad_value = pad_values[key]

                if _length_dim != feature.dim() - 1:
                    feature = feature.transpose(_length_dim, -1)

                sliced_feature = F.pad(feature, (0, padding), value=pad_value)

                if _length_dim != feature.dim() - 1:
                    sliced_feature = sliced_feature.transpose(_length_dim, -1)

                low_start_idx = 0
        else:
            if random_slice:
                if length == sliced_feature_length:
                    low_start_idx = 0
                else:
                    low_start_idx = torch.randint(
                        0, length - sliced_feature_length, (), dtype=torch.long
                    ).item()
            else:
                low_start_idx = length // 2 - sliced_feature_length // 2

            low_end_idx = low_start_idx + sliced_feature_length

            _, sliced_feature, _ = torch.split(
                feature,
                [
                    low_start_idx,
                    low_end_idx - low_start_idx,
                    feature.size(_length_dim) - low_end_idx,
                ],
                dim=_length_dim,
            )

        slice_key = key_mapping[key]
        output_batch[slice_key].append(sliced_feature)

        for key in key_mapping.keys():
            if key == low_resolution_key:
                continue

            feature = output_batch[key][sample_idx]
            length_key = _length_mapping[key]
            length_dim = _length_dims[key]
            hop_length = hop_lengths[key]
            sliced_feature_length = math.ceil(slice_length / hop_length)
            _length_dim = _compute_dim_without_batch(length_dim)

            if _length_dim < 0:
                # set _length_dim to be non-negative
                _length_dim = feature.dim() + _length_dim

            length = _compute_length(
                output_batch,
                key,
                sample_idx,
                length_key=length_key,
                length_dim=_length_dim,
            )

            if length < sliced_feature_length:
                # remove exising padding
                padding = feature.size(_length_dim) - length
                feature, _ = torch.split(
                    feature, [length, feature.size(_length_dim) - length], dim=_length_dim
                )
                # append padding
                padding = sliced_feature_length - length
                pad_value = pad_values[key]

                if _length_dim != feature.dim() - 1:
                    feature = feature.transpose(_length_dim, -1)

                sliced_feature = F.pad(feature, (0, padding), value=pad_value)

                if _length_dim != feature.dim() - 1:
                    sliced_feature = sliced_feature.transpose(_length_dim, -1)
            else:
                start_idx = low_start_idx * hop_lengths[low_resolution_key]
                start_idx = start_idx // hop_length
                end_idx = start_idx + sliced_feature_length
                slice_key = key_mapping[key]

                _, sliced_feature, _ = torch.split(
                    feature,
                    [start_idx, end_idx - start_idx, feature.size(_length_dim) - end_idx],
                    dim=_length_dim,
                )

            output_batch[slice_key].append(sliced_feature)

    for slice_key in key_mapping.values():
        output_batch[slice_key] = torch.stack(output_batch[slice_key], dim=0)

    return output_batch


def take_log_features(
    batch: Dict[str, Any],
    key_mapping: Optional[Dict[str, str]] = None,
    flooring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Add log-features from given batch.

    Args:
        batch (dict): Dict-type batch.
        key_mapping (dict, optional): Mapping of keys to log-feature.

    Returns:
        dict: Dict-type batch including log-features.

    """
    if key_mapping is None:
        key_mapping = {}

    for key, log_key in key_mapping.items():
        if flooring_fn is None:
            feature = batch[key]
        else:
            feature = flooring_fn(batch[key])

        batch[log_key] = torch.log(feature)

    return batch


def make_noise(
    batch: Dict[str, Any],
    key_mapping: Optional[Dict[str, str]] = None,
    std: float = 1,
) -> Dict[str, Any]:
    """Make noise from given batch.

    Args:
        batch (dict): Dict-type batch.
        key_mapping (dict, optional): Mapping of keys to make noise.
        std (float or dict): Noise scael. Default: ``1``.

    Returns:
        dict: Dict-type batch including noise.

    """
    if key_mapping is None:
        key_mapping = {}

    if not isinstance(std, dict):
        std = {key: std for key in key_mapping.keys()}

    for key, noise_key in key_mapping.items():
        batch[noise_key] = std[key] * torch.randn_like(batch[key])

    return batch


def _compute_length(
    batch: Dict[str, Any],
    key: str,
    sample_idx: int,
    length_key: Optional[str] = None,
    length_dim: Optional[int] = None,
) -> int:
    """Compute length after removing padding."""
    if length_dim is None:
        length_dim = -1

    feature: torch.Tensor = batch[key][sample_idx]

    if length_key is None:
        length = feature.size(length_dim)
    else:
        length = batch[length_key][sample_idx].item()

    return length
