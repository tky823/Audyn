import torch

from ..._cpp_extensions import monotonic_align as monotonic_align_cpp

__all__ = ["viterbi_monotonic_alignment"]


@torch.no_grad()
def viterbi_monotonic_alignment(
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
    hard_alignment = monotonic_align_cpp.viterbi_monotonic_alignment(probs, take_log)

    return hard_alignment
