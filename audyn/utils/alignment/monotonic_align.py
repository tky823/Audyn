from typing import Optional

import torch

from ..._cpp_extensions import monotonic_align as monotonic_align_cpp

__all__ = ["search_monotonic_alignment_by_viterbi"]


@torch.no_grad()
def search_monotonic_alignment_by_viterbi(
    probs: torch.Tensor,
    padding_mask: Optional[torch.BoolTensor] = None,
    take_log: bool = False,
) -> torch.LongTensor:
    """Search monotonic alignment by Viterbi algorithm.

    Args:
        probs (torch.Tensor): Soft alignment matrix of shape (batch_size, tgt_length, src_length).
        padding_mask (torch.Tensor): Padding mask to ignore padding grids.
            The shape is (batch_size, tgt_length, src_length).
        take_log (bool): If ``True``, log of ``probs`` is treated as log probability.
            Otherwise, ``probs`` is treated as log probability. Default: ``False``.

    Returns:
        torch.LongTensor: Hard alignment matrix of shape (batch_size, tgt_length, src_length).

    .. note::

        Monotonic alignment search is always computed on CPU.

    """
    src_device = probs.device
    probs = probs.cpu()

    if padding_mask is not None:
        padding_mask = padding_mask.cpu()

    if padding_mask is not None:
        if take_log:
            # Set p(x) as 0.
            probs = probs.masked_fill(padding_mask, 0)
        else:
            # Set log(p(x)) as -inf.
            probs = probs.masked_fill(padding_mask, -float("inf"))

        non_padding_mask = torch.logical_not(padding_mask).long()

        sum_non_padding_mask = torch.sum(non_padding_mask, dim=2)
        tgt_length = torch.count_nonzero(sum_non_padding_mask, dim=1)

        sum_non_padding_mask = torch.sum(non_padding_mask, dim=1)
        src_length = torch.count_nonzero(sum_non_padding_mask, dim=1)
    else:
        batch_size, max_tgt_length, max_src_length = probs.size()
        factory_kwargs = {"device": probs.device, "dtype": torch.long}

        tgt_length = torch.full((batch_size,), max_tgt_length, **factory_kwargs)
        src_length = torch.full((batch_size,), max_src_length, **factory_kwargs)

    hard_alignment = monotonic_align_cpp.search_monotonic_alignment_by_viterbi(
        probs, tgt_length, src_length, take_log
    )
    hard_alignment = hard_alignment.to(src_device)

    return hard_alignment
