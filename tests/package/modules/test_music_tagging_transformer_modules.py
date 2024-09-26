import pytest
import torch

from audyn.modules.music_tagging_transformer import (
    MusicTaggingTransformerEncoder,
    PositionalPatchEmbedding,
)


def test_music_tagging_transformer_positional_patch_embedding() -> None:
    out_channels, hidden_channels = 16, 32
    kernel_size = 3

    batch_size = 4
    n_bins, n_frames = 32, 50

    module = PositionalPatchEmbedding(
        out_channels, hidden_channels, n_bins, kernel_size=kernel_size
    )

    input = torch.randn((batch_size, n_bins, n_frames))
    output = module(input)

    assert output.size(0) == batch_size
    assert output.size(-1) == out_channels


@pytest.mark.parametrize("batch_first", [True, False])
def test_music_tagging_transformer_encoder(batch_first: bool) -> None:
    d_model = 16
    nhead = 2
    num_layers = 2
    activation = "gelu"

    batch_size = 4
    max_frames = 50

    encoder = MusicTaggingTransformerEncoder(
        d_model,
        nhead,
        num_layers=num_layers,
        dim_feedforward=4 * d_model,
        activation=activation,
        batch_first=batch_first,
    )

    lengths = torch.randint(max_frames // 2, max_frames, (batch_size,))
    input = torch.randn((max_frames, batch_size, d_model))

    if batch_first:
        input = torch.randn((batch_size, max_frames, d_model))
    else:
        input = torch.randn((max_frames, batch_size, d_model))

    output = encoder(input)

    assert output.size() == input.size()

    padding_mask = torch.arange(max_frames) >= lengths.unsqueeze(dim=-1)

    if batch_first:
        input = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
    else:
        _padding_mask = padding_mask.transpose(1, 0)
        input = input.masked_fill(_padding_mask.unsqueeze(dim=-1), 0)

    output = encoder(input, src_key_padding_mask=padding_mask)

    assert output.size() == input.size()
