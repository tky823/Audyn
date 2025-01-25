import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.modules.swintransformer import (
    SwinRelativePositionalMultiheadAttention,
    SwinTransformerEncoderLayer,
)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
def test_swin_transformer_encoder_layer(bias: bool, batch_first: bool, share_heads: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    embed_dim = 8
    dim_feedforward = 16
    num_heads = 4
    window_height, window_width = 5, 7
    height, width = 4 * window_height, 4 * window_width

    if batch_first:
        input = torch.randn((batch_size, height * width, embed_dim))
    else:
        input = torch.randn((height * width, batch_size, embed_dim))

    model = SwinTransformerEncoderLayer(
        embed_dim,
        num_heads,
        dim_feedforward=dim_feedforward,
        bias=bias,
        height=height,
        width=width,
        window_size=(window_height, window_width),
        share_heads=share_heads,
        batch_first=batch_first,
    )

    output = model(input)

    if batch_first:
        assert output.size() == (batch_size, height * width, embed_dim)
    else:
        assert output.size() == (height * width, batch_size, embed_dim)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
def test_swin_relative_positional_attn(bias: bool, batch_first: bool, share_heads: bool) -> None:
    torch.manual_seed(0)

    batch_size = 3
    embed_dim = 8
    num_heads = 4
    window_height, window_width = 5, 7

    if batch_first:
        input = torch.randn((batch_size, window_height * window_width, embed_dim))
    else:
        input = torch.randn((window_height * window_width, batch_size, embed_dim))

    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )
    relative_mha = SwinRelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        window_size=(window_height, window_width),
        share_heads=share_heads,
        batch_first=batch_first,
    )

    relative_mha.in_proj_weight.data.copy_(mha.in_proj_weight.data)
    relative_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)

    if bias:
        assert mha.in_proj_bias is not None
        assert mha.out_proj.bias is not None

        relative_mha.in_proj_bias.data.copy_(mha.in_proj_bias.data)
        relative_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)
    else:
        assert mha.in_proj_bias is None
        assert mha.out_proj.bias is None
        assert relative_mha.in_proj_bias is None
        assert relative_mha.out_proj.bias is None

    output, attn_weights = mha(input, input, input)
    relative_output, relative_attn_weights = relative_mha(input, input, input)

    if batch_first:
        assert relative_output.size() == (batch_size, window_height * window_width, embed_dim)
    else:
        assert relative_output.size() == (window_height * window_width, batch_size, embed_dim)

    assert relative_attn_weights.size() == (
        batch_size,
        window_height * window_width,
        window_height * window_width,
    )

    allclose(output, relative_output, atol=1e-7)
    allclose(attn_weights, relative_attn_weights)
