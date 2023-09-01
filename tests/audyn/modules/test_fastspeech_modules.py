import pytest
import torch

from audyn.modules.fastspeech import ConvBlock, FFTrBlock, MultiheadAttentionBlock

parameters_batch_first = [True, False]


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_fftr_block(batch_first: bool):
    batch_size, max_length = 4, 16
    d_model, hidden_channels = 8, 2
    num_heads = 2
    kernel_size = 3

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, max_length, d_model))
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(src_key_padding_mask.unsqueeze(dim=-1), 0)

    if not batch_first:
        input = input.permute(1, 0, 2)

    module = FFTrBlock(
        d_model,
        hidden_channels=hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    output, attn_weights = module(input)

    if batch_first:
        assert output.size() == (batch_size, max_length, d_model)
    else:
        assert output.size() == (max_length, batch_size, d_model)

    assert attn_weights.size() == (batch_size, max_length, max_length)

    output, attn_weights = module(input, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        assert output.size() == (batch_size, max_length, d_model)
    else:
        assert output.size() == (max_length, batch_size, d_model)

    assert attn_weights.size() == (batch_size, max_length, max_length)


def test_conv_block():
    batch_size, max_length = 4, 16
    num_features, hidden_channels = 8, 2
    kernel_size = 3

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)

    module = ConvBlock(num_features, hidden_channels, kernel_size=kernel_size)
    output = module(input)

    assert output.size() == (batch_size, num_features, max_length)

    output = module(input, padding_mask=padding_mask)

    assert output.size() == (batch_size, num_features, max_length)


def test_mha_block():
    batch_size, max_length = 4, 16
    embed_dim = 8
    num_heads = 2

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, max_length, embed_dim))
    time_seq = torch.arange(max_length)
    key_padding_mask = time_seq >= length.unsqueeze(dim=-1)
    causal_mask = time_seq > time_seq.unsqueeze(dim=-1)

    module = MultiheadAttentionBlock(embed_dim, num_heads=num_heads, batch_first=True)
    output, attn_weights = module(input)

    assert output.size() == (batch_size, max_length, embed_dim)
    assert attn_weights.size() == (batch_size, max_length, max_length)

    output, attn_weights = module(input, key_padding_mask=key_padding_mask, attn_mask=causal_mask)

    assert output.size() == (batch_size, max_length, embed_dim)
    assert attn_weights.size() == (batch_size, max_length, max_length)
