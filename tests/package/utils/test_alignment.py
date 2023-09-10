import pytest
import torch
import torch.nn.functional as F

from audyn.utils.alignment.monotonic_align import viterbi_monotonic_alignment


@pytest.mark.parametrize("take_log", [True, False])
def test_viterbi_monotonic_alignment(take_log: bool) -> None:
    torch.manual_seed(0)

    batch_size = 4
    max_src_length = 10
    max_tgt_length = 2 * max_src_length
    tgt_lengths = torch.randint(max_tgt_length // 2, max_tgt_length + 1, (batch_size,))
    src_lengths = torch.randint(max_src_length // 2, max_src_length + 1, (batch_size,))
    max_tgt_length, max_src_length = torch.max(tgt_lengths), torch.max(src_lengths)
    log_probs = torch.randn((batch_size, max_tgt_length, max_src_length))
    tgt_indices, src_indices = torch.arange(max_tgt_length), torch.arange(max_src_length)
    padding_mask = src_indices >= src_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
    log_probs = log_probs.masked_fill(padding_mask, -float("inf"))
    probs = torch.softmax(log_probs, dim=-1)
    padding_mask = tgt_indices.unsqueeze(dim=-1) >= tgt_lengths.unsqueeze(dim=-1).unsqueeze(dim=-1)
    probs = probs.masked_fill(padding_mask, 0)

    if not take_log:
        probs = torch.log(probs)

    hard_alignments = viterbi_monotonic_alignment(probs, take_log=take_log)

    for hard_alignment, tgt_length, src_length in zip(hard_alignments, tgt_lengths, src_lengths):
        hard_alignment = F.pad(
            hard_alignment,
            (0, src_length.item() - max_src_length, 0, tgt_length.item() - max_tgt_length),
        )
        assert torch.all(hard_alignment.sum(dim=1) == 1)
