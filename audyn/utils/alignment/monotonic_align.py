import torch

from ..._cpp_extensions import monotonic_align as monotonic_align_cpp

__all__ = ["search_monotonic_alignment_by_viterbi"]


@torch.no_grad()
def search_monotonic_alignment_by_viterbi(
    probs: torch.Tensor,
    take_log: bool = False,
) -> torch.LongTensor:
    """Search monotonic alignment by Viterbi algorithm.

    Args:
        probs (torch.Tensor): Soft alignment matrix of shape (batch_size, tgt_length, src_length).
        take_log (bool): If ``True``, log of ``probs`` is treated as log probability.
            Otherwise, ``probs`` is treated as log probability. Default: ``False``.

    Returns:
        torch.LongTensor: Hard alignment matrix of shape (batch_size, tgt_length, src_length).

    """
    hard_alignment = monotonic_align_cpp.search_monotonic_alignment_by_viterbi(probs, take_log)

    return hard_alignment
