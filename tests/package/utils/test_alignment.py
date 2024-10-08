import pytest
import torch
import torch.nn.functional as F

from audyn.utils.alignment import expand_by_duration
from audyn.utils.alignment.monotonic_align import search_monotonic_alignment_by_viterbi


@pytest.mark.parametrize("feature_dim", [0, 1])
def test_expand_by_duration(feature_dim: int) -> None:
    src = torch.tensor(
        [
            [[1, -1], [2, -2], [3, -3], [0, 0]],
            [[4, -4], [5, -5], [0, 0], [0, 0]],
            [[6, -6], [7, -7], [8, -8], [9, -9]],
        ]
    )
    duration = torch.tensor([[3, 4, 2, 0], [2, 1, 0, 0], [1, 1, 3, 2]])

    tgt = torch.tensor(
        [
            [[1, -1], [1, -1], [1, -1], [2, -2], [2, -2], [2, -2], [2, -2], [3, -3], [3, -3]],
            [[4, -4], [4, -4], [5, -5], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
            [[6, -6], [7, -7], [8, -8], [8, -8], [8, -8], [9, -9], [9, -9], [0, 0], [0, 0]],
        ]
    )

    if feature_dim > 0:
        # to test (batch_size, src_length, *shape) as sequence
        pass
    else:
        # to test (batch_size, src_length) as sequence
        src = src[..., 0]
        tgt = tgt[..., 0]

    est = expand_by_duration(src, duration, batch_first=True)

    assert torch.all(est == tgt)

    max_length = tgt.size(1) + 1

    if feature_dim > 0:
        tgt = F.pad(tgt, (0, 0, 0, 1))
    else:
        tgt = F.pad(tgt, (0, 1))

    est = expand_by_duration(src, duration, batch_first=True, max_length=max_length)

    assert torch.all(est == tgt)


@pytest.mark.parametrize("take_log", [True, False])
def test_search_monotonic_alignment_by_viterbi(take_log: bool) -> None:
    torch.manual_seed(0)

    batch_size = 4
    max_src_length = 10
    max_tgt_length = 2 * max_src_length
    tgt_lengths = torch.randint(max_tgt_length // 2, max_tgt_length + 1, (batch_size,))
    src_lengths = torch.randint(max_src_length // 2, max_src_length + 1, (batch_size,))
    max_tgt_length, max_src_length = torch.max(tgt_lengths), torch.max(src_lengths)
    log_probs = torch.randn((batch_size, max_tgt_length, max_src_length))
    tgt_indices, src_indices = torch.arange(max_tgt_length), torch.arange(max_src_length)
    src_padding_mask = src_indices >= src_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
    log_probs = log_probs.masked_fill(src_padding_mask, -float("inf"))
    probs = torch.softmax(log_probs, dim=-1)
    tgt_padding_mask = tgt_indices.unsqueeze(dim=-1) >= tgt_lengths.unsqueeze(dim=-1).unsqueeze(
        dim=-1
    )
    probs = probs.masked_fill(tgt_padding_mask, 0)
    padding_mask = src_padding_mask | tgt_padding_mask

    if not take_log:
        probs = torch.log(probs)

    hard_alignments = search_monotonic_alignment_by_viterbi(
        probs, padding_mask=padding_mask, take_log=take_log
    )

    for hard_alignment, tgt_length, src_length in zip(hard_alignments, tgt_lengths, src_lengths):
        hard_alignment = F.pad(
            hard_alignment,
            (0, src_length.item() - max_src_length, 0, tgt_length.item() - max_tgt_length),
        )
        assert torch.all(hard_alignment.sum(dim=1) == 1)
