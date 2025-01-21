import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.modules.swintransformer import SwinRelativePositionalMultiheadAttention


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
def test_swin_relative_positional_attn(bias: bool, batch_first: bool, share_heads: bool) -> None:
    torch.manual_seed(0)

    batch_size = 3
    height, width = 5, 7
    embed_dim = 8
    num_heads = 4

    if batch_first:
        input = torch.randn((batch_size, height * width, embed_dim))
    else:
        input = torch.randn((height * width, batch_size, embed_dim))

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
        window_size=(height, width),
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
        assert relative_output.size() == (batch_size, height * width, embed_dim)
    else:
        assert relative_output.size() == (height * width, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, height * width, height * width)

    allclose(output, relative_output, atol=1e-7)
    allclose(attn_weights, relative_attn_weights)
