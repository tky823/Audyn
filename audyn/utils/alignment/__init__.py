from typing import Optional

import torch

__all__ = ["expand_by_duration"]


def expand_by_duration(
    sequence: torch.Tensor,
    duration: torch.LongTensor,
    pad_value: float = 0,
    batch_first: bool = True,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    r"""Expand sequence by given duration.

    Args:
        sequence (torch.Tensor): Latent feature of source whose shape is
            (batch_size, src_length, \*) if ``batch_first=True``. Otherwise,
            the shape is regarded as (src_length, batch_size, \*).
        duration (torch.LongTensor): Duration of shape (batch_size, src_length). if
            ``batch_first=True``. Otherwise, the shape is regarded as (src_length, batch_size).
        pad_value (float): Padding value.
        batch_first (bool): Parameter to determine shape of input.
        max_length (int): Max length of sum of target durations.

    Returns:
        torch.Tensor: Expanded latent feature of shape (batch_size, max_tgt_length, \*).

    .. note::

        (batch_size, src_length) and (src_length, batch_size) are also supported as ``sequence``.

    """
    if not batch_first:
        sequence = sequence.swapaxes(0, 1)
        duration = duration.swapaxes(0, 1)

    batch_size = sequence.size(0)
    feature_shape = sequence.size()[2:]

    if max_length is None:
        max_tgt_length = torch.max(duration.sum(dim=1))
        max_tgt_length = max_tgt_length.item()
    else:
        max_tgt_length = max_length

    expanded = torch.full(
        (batch_size, max_tgt_length, *feature_shape),
        fill_value=pad_value,
        dtype=sequence.dtype,
        device=sequence.device,
    )

    for batch_idx in range(batch_size):
        x = torch.repeat_interleave(sequence[batch_idx], duration[batch_idx], dim=0)
        d = len(x)
        expanded[batch_idx, :d] = x

    if not batch_first:
        expanded = expanded.swapaxes(0, 1)

    return expanded
