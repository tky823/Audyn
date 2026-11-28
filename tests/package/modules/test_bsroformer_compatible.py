import pytest
import torch
import torch.nn as nn

from audyn.modules.bsrnn import (
    AudioSeparatorCompatibleBandMergeModule,
    AudioSeparatorCompatibleBandSplitModule,
)
from audyn.modules.bsroformer import (
    AudioSeparatorCompatibleRoFormerEncoderLayer,
    AudioSeparatorCompatibleRotaryPositionalMultiheadAttention,
)


@pytest.mark.parametrize("bins", [[2, 2], [[0, 2], [1, 3]]])
def test_audio_separator_band_modules_forward(bins) -> None:
    torch.manual_seed(0)
    bandsplit = AudioSeparatorCompatibleBandSplitModule(1, bins, 8, norm="rms")
    bandmerge = AudioSeparatorCompatibleBandMergeModule(
        1, bins, 8, hidden_channels=12, norm="layer_norm"
    )
    spectrogram = torch.randn(2, 1, 4 if isinstance(bins[0], int) else 3, 5).to(torch.complex64)

    features = bandsplit(spectrogram)
    output = bandmerge(features)

    assert features.shape == (2, 8, 2, 5)
    assert output.shape == spectrogram.shape
    assert torch.isfinite(output).all()


def test_audio_separator_band_merge_averages_overlapping_bands() -> None:
    bins = [[0, 2], [1, 3]]
    bandmerge = AudioSeparatorCompatibleBandMergeModule(1, bins, 4, hidden_channels=8)

    class ConstantBand(nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return torch.full((input.size(0), 1, 2, input.size(-1)), self.value)

    bandmerge.backbone = nn.ModuleList([ConstantBand(2), ConstantBand(4)])
    output = bandmerge(torch.zeros(1, 4, 2, 3))

    torch.testing.assert_close(output[0, 0], torch.tensor([[2.0] * 3, [3.0] * 3, [4.0] * 3]))


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("separate_qkv", [True, False])
def test_audio_separator_attention_weights_and_gradients(batch_first, separate_qkv) -> None:
    torch.manual_seed(0)
    attention = AudioSeparatorCompatibleRotaryPositionalMultiheadAttention(
        8, 2, qdim=6 if separate_qkv else None, batch_first=batch_first
    )
    shape = (2, 3, 6 if separate_qkv else 8)
    query = torch.randn(shape if batch_first else (3, 2, shape[-1]), requires_grad=True)
    key = torch.randn_like(query)
    mask = torch.tensor([[False, False, True], [False, False, True]])

    output, weights = attention(query, key, key, key_padding_mask=mask)
    output.sum().backward()

    assert output.shape == query.shape
    assert weights.shape == (2, 3, 3)
    torch.testing.assert_close(weights[..., 2], torch.zeros_like(weights[..., 2]))
    assert query.grad is not None and torch.isfinite(query.grad).all()

    output, weights = attention(query, key, key, need_weights=False)
    assert output.shape == query.shape
    assert weights is None

    # Self-attention and separate key/value inputs take different projection paths.
    output, weights = attention(query, query, query, average_attn_weights=False)
    assert output.shape == query.shape
    assert weights.shape == (2, 2, 3, 3)
    output, weights = attention(query, key, torch.randn_like(key), average_attn_weights=False)
    assert output.shape == query.shape
    assert weights.shape == (2, 2, 3, 3)


@pytest.mark.parametrize("norm,norm_first", [("rms", True), ("layer_norm", False)])
def test_audio_separator_encoder_norm_paths(norm, norm_first) -> None:
    torch.manual_seed(0)
    layer = AudioSeparatorCompatibleRoFormerEncoderLayer(
        8, 2, dim_feedforward=12, norm=norm, norm_first=norm_first, batch_first=True
    )
    input = torch.randn(2, 3, 8, requires_grad=True)

    output = layer(input)
    output.square().mean().backward()

    assert output.shape == input.shape
    assert input.grad is not None and torch.isfinite(input.grad).all()


def test_audio_separator_encoder_rejects_causal_attention() -> None:
    layer = AudioSeparatorCompatibleRoFormerEncoderLayer(8, 2, batch_first=True)

    with pytest.raises(NotImplementedError, match="is_causal=True"):
        layer(torch.randn(1, 3, 8), is_causal=True)


@pytest.mark.parametrize("module", ["split", "merge"])
@pytest.mark.parametrize("norm", ["unsupported", 1])
def test_audio_separator_band_modules_reject_invalid_norm(module, norm) -> None:
    with pytest.raises(ValueError, match="Unsupported norm type"):
        if module == "split":
            AudioSeparatorCompatibleBandSplitModule(1, [2], 8, norm=norm)
        else:
            AudioSeparatorCompatibleBandMergeModule(1, [2], 8, norm=norm)
