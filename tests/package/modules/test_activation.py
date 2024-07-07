from typing import Optional, Tuple

import pytest
import torch
import torch.nn as nn

from audyn.modules.activation import (
    ExtrapolatablePositionalMultiheadAttention,
    RelativePositionalMultiheadAttention,
    RotaryPositionalMultiheadAttention,
    TrainableAbsolutePositionalMultiheadAttention,
    TransformerXLRelativePositionalMultiheadAttention,
)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
@pytest.mark.parametrize("need_weights", [True, False])
def test_trainable_absolute_positional_attn(
    bias: bool, batch_first: bool, use_attn_mask: bool, share_heads: bool, need_weights: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_pos_length, max_query_length, max_key_length = 16, 12, 10
    embed_dim, kdim, vdim = 8, 4, 5
    num_heads = 4

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size, max_query_length, max_key_length, embed_dim, batch_first=batch_first
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    absolute_mha = TrainableAbsolutePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        max_length=max_pos_length,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    absolute_output, absolute_attn_weights = absolute_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert absolute_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert absolute_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert absolute_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    # compatibility with nn.MultiheadAttention
    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    absolute_mha.q_positional_embedding.data.zero_()
    absolute_mha.k_positional_embedding.data.zero_()
    absolute_mha.v_positional_embedding.data.zero_()

    absolute_mha.in_proj_weight.data.copy_(mha.in_proj_weight.data)
    absolute_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)

    if bias:
        assert absolute_mha.in_proj_bias is not None
        assert mha.in_proj_bias is not None
        assert absolute_mha.out_proj.bias is not None
        assert mha.out_proj.bias is not None

        absolute_mha.in_proj_bias.data.copy_(mha.in_proj_bias.data)
        absolute_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)
    else:
        assert absolute_mha.in_proj_bias is None
        assert mha.in_proj_bias is None
        assert absolute_mha.out_proj.bias is None
        assert mha.out_proj.bias is None

    output, attn_weights = mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )
    absolute_output, absolute_attn_weights = absolute_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert absolute_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert absolute_output.size() == (max_query_length, batch_size, embed_dim)

    assert torch.allclose(output, absolute_output, atol=1e-7)

    if need_weights:
        assert absolute_attn_weights.size() == (batch_size, max_query_length, max_key_length)
        assert torch.allclose(attn_weights, absolute_attn_weights)

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        embed_dim,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    absolute_mha = TrainableAbsolutePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        max_length=max_pos_length,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    absolute_output, absolute_attn_weights = absolute_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert absolute_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert absolute_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert absolute_attn_weights.size() == (batch_size, max_query_length, max_key_length)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize("longer_window", [True, False])
def test_relative_positional_attn(
    bias: bool, batch_first: bool, use_attn_mask: bool, longer_window: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_query_length, max_key_length = 12, 10
    embed_dim, kdim, vdim = 8, 4, 5
    num_heads = 4

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size, max_query_length, max_key_length, embed_dim, batch_first=batch_first
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    if longer_window:
        window_size = max(max_query_length, max_key_length)
    else:
        window_size = min(max_query_length, max_key_length) - 2

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    relative_mha = RelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        window_size=window_size,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert relative_output.size() == (max_query_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    # compatibility with nn.MultiheadAttention
    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    relative_mha.k_positional_embedding.data.zero_()
    relative_mha.v_positional_embedding.data.zero_()

    relative_mha.in_proj_weight.data.copy_(mha.in_proj_weight.data)
    relative_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)

    if bias:
        assert relative_mha.in_proj_bias is not None
        assert mha.in_proj_bias is not None
        assert relative_mha.out_proj.bias is not None
        assert mha.out_proj.bias is not None

        relative_mha.in_proj_bias.data.copy_(mha.in_proj_bias.data)
        relative_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)
    else:
        assert relative_mha.in_proj_bias is None
        assert mha.in_proj_bias is None
        assert relative_mha.out_proj.bias is None
        assert mha.out_proj.bias is None

    output, attn_weights = mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )
    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert relative_output.size() == (max_query_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    assert torch.allclose(output, relative_output, atol=1e-7)
    assert torch.allclose(attn_weights, relative_attn_weights)

    # ensure invariance of relative positions
    relative_mha = RelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        window_size=window_size,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        random_padding = torch.randn((batch_size, 1, embed_dim))
        query = torch.cat([random_padding, query], dim=1)
        key = torch.cat([random_padding, key], dim=1)
        value = torch.cat([random_padding, value], dim=1)
    else:
        random_padding = torch.randn((1, batch_size, embed_dim))
        query = torch.cat([random_padding, query], dim=0)
        key = torch.cat([random_padding, key], dim=0)
        value = torch.cat([random_padding, value], dim=0)

    padding = torch.full((batch_size, 1), fill_value=True)
    key_padding_mask = torch.cat([padding, key_padding_mask], dim=1)

    if attn_mask is None:
        # Even when attn_mask is None, positions of key_padding_mask will be ignored
        # in attention weights.
        pass
    else:
        padding = torch.full((max_query_length, 1), fill_value=True)
        attn_mask = torch.cat([padding, attn_mask], dim=-1)
        padding = torch.full((1, max_key_length + 1), fill_value=False)
        attn_mask = torch.cat([padding, attn_mask], dim=-2)

    padded_relative_output, padded_relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        _, padded_relative_output = torch.split(
            padded_relative_output, [1, max_query_length], dim=1
        )
    else:
        _, padded_relative_output = torch.split(
            padded_relative_output, [1, max_query_length], dim=0
        )

    _, padded_relative_attn_weights = torch.split(
        padded_relative_attn_weights, [1, max_query_length], dim=-2
    )
    _, padded_relative_attn_weights = torch.split(
        padded_relative_attn_weights, [1, max_key_length], dim=-1
    )

    assert torch.allclose(padded_relative_output, relative_output, atol=1e-7)
    assert torch.allclose(padded_relative_attn_weights, relative_attn_weights)

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        embed_dim,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    if longer_window:
        window_size = max(max_query_length, max_key_length)
    else:
        window_size = min(max_query_length, max_key_length) - 2

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    relative_mha = RelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        window_size=window_size,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert relative_output.size() == (max_query_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_query_length, max_key_length)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
def test_transformer_xl_relative_positional_attn(
    bias: bool, batch_first: bool, use_attn_mask: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_query_length, max_key_length = 12, 10
    embed_dim, kdim, vdim = 8, 4, 5
    num_heads = 4

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size, max_query_length, max_key_length, embed_dim, batch_first=batch_first
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    relative_mha = TransformerXLRelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert relative_output.size() == (max_query_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    # ensure invariance of relative positions
    relative_mha = TransformerXLRelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        random_padding = torch.randn((batch_size, 1, embed_dim))
        query = torch.cat([random_padding, query], dim=1)
        key = torch.cat([random_padding, key], dim=1)
        value = torch.cat([random_padding, value], dim=1)
    else:
        random_padding = torch.randn((1, batch_size, embed_dim))
        query = torch.cat([random_padding, query], dim=0)
        key = torch.cat([random_padding, key], dim=0)
        value = torch.cat([random_padding, value], dim=0)

    padding = torch.full((batch_size, 1), fill_value=True)
    key_padding_mask = torch.cat([padding, key_padding_mask], dim=1)

    if attn_mask is None:
        # Even when attn_mask is None, positions of key_padding_mask will be ignored
        # in attention weights.
        pass
    else:
        padding = torch.full((max_query_length, 1), fill_value=True)
        attn_mask = torch.cat([padding, attn_mask], dim=-1)
        padding = torch.full((1, max_key_length + 1), fill_value=False)
        attn_mask = torch.cat([padding, attn_mask], dim=-2)

    padded_relative_output, padded_relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        _, padded_relative_output = torch.split(
            padded_relative_output, [1, max_query_length], dim=1
        )
    else:
        _, padded_relative_output = torch.split(
            padded_relative_output, [1, max_query_length], dim=0
        )

    _, padded_relative_attn_weights = torch.split(
        padded_relative_attn_weights, [1, max_query_length], dim=-2
    )
    _, padded_relative_attn_weights = torch.split(
        padded_relative_attn_weights, [1, max_key_length], dim=-1
    )

    assert torch.allclose(padded_relative_output, relative_output)
    assert torch.allclose(padded_relative_attn_weights, relative_attn_weights)

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        embed_dim,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    relative_mha = TransformerXLRelativePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert relative_output.size() == (max_query_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_query_length, max_key_length)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
@pytest.mark.parametrize("need_weights", [True, False])
def test_rotary_positional_attn(
    bias: bool, batch_first: bool, use_attn_mask: bool, share_heads: bool, need_weights: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_query_length, max_key_length = 12, 10
    embed_dim, kdim, vdim = 8, 4, 5
    num_heads = 4

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size, max_query_length, max_key_length, embed_dim, batch_first=batch_first
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    rotary_mha = RotaryPositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    rotary_output, rotary_attn_weights = rotary_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert rotary_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert rotary_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert rotary_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    # ensure invariance of relative positions
    if batch_first:
        random_padding = torch.randn((batch_size, 1, embed_dim))
        query = torch.cat([random_padding, query], dim=1)
        key = torch.cat([random_padding, key], dim=1)
        value = torch.cat([random_padding, value], dim=1)
    else:
        random_padding = torch.randn((1, batch_size, embed_dim))
        query = torch.cat([random_padding, query], dim=0)
        key = torch.cat([random_padding, key], dim=0)
        value = torch.cat([random_padding, value], dim=0)

    padding = torch.full((batch_size, 1), fill_value=True)
    key_padding_mask = torch.cat([padding, key_padding_mask], dim=1)

    if attn_mask is None:
        # Even when attn_mask is None, positions of key_padding_mask will be ignored
        # in attention weights.
        pass
    else:
        padding = torch.full((max_query_length, 1), fill_value=True)
        attn_mask = torch.cat([padding, attn_mask], dim=-1)
        padding = torch.full((1, max_key_length + 1), fill_value=False)
        attn_mask = torch.cat([padding, attn_mask], dim=-2)

    padded_rotary_output, padded_rotary_attn_weights = rotary_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        _, padded_rotary_output = torch.split(padded_rotary_output, [1, max_query_length], dim=1)
    else:
        _, padded_rotary_output = torch.split(padded_rotary_output, [1, max_query_length], dim=0)

    assert torch.allclose(padded_rotary_output, rotary_output, atol=1e-7)

    if need_weights:
        _, padded_rotary_attn_weights = torch.split(
            padded_rotary_attn_weights, [1, max_query_length], dim=-2
        )
        _, padded_rotary_attn_weights = torch.split(
            padded_rotary_attn_weights, [1, max_key_length], dim=-1
        )

        assert torch.allclose(padded_rotary_attn_weights, rotary_attn_weights, atol=1e-6)

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        embed_dim,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    rotary_mha = RotaryPositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    rotary_output, rotary_attn_weights = rotary_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert rotary_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert rotary_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert rotary_attn_weights.size() == (batch_size, max_query_length, max_key_length)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
@pytest.mark.parametrize("need_weights", [True, False])
def test_extrapolatable_positional_attn(
    bias: bool, batch_first: bool, use_attn_mask: bool, share_heads: bool, need_weights: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_query_length, max_key_length = 12, 10
    embed_dim, kdim, vdim = 8, 4, 5
    num_heads = 4

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size, max_query_length, max_key_length, embed_dim, batch_first=batch_first
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    xpos_mha = ExtrapolatablePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    xpos_output, xpos_attn_weights = xpos_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert xpos_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert xpos_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert xpos_attn_weights.size() == (batch_size, max_query_length, max_key_length)

    # ensure invariance of relative positions
    if batch_first:
        random_padding = torch.randn((batch_size, 1, embed_dim))
        query = torch.cat([random_padding, query], dim=1)
        key = torch.cat([random_padding, key], dim=1)
        value = torch.cat([random_padding, value], dim=1)
    else:
        random_padding = torch.randn((1, batch_size, embed_dim))
        query = torch.cat([random_padding, query], dim=0)
        key = torch.cat([random_padding, key], dim=0)
        value = torch.cat([random_padding, value], dim=0)

    padding = torch.full((batch_size, 1), fill_value=True)
    key_padding_mask = torch.cat([padding, key_padding_mask], dim=1)

    if attn_mask is None:
        # Even when attn_mask is None, positions of key_padding_mask will be ignored
        # in attention weights.
        pass
    else:
        padding = torch.full((max_query_length, 1), fill_value=True)
        attn_mask = torch.cat([padding, attn_mask], dim=-1)
        padding = torch.full((1, max_key_length + 1), fill_value=False)
        attn_mask = torch.cat([padding, attn_mask], dim=-2)

    padded_xpos_output, padded_xpos_attn_weights = xpos_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        _, padded_xpos_output = torch.split(padded_xpos_output, [1, max_query_length], dim=1)
    else:
        _, padded_xpos_output = torch.split(padded_xpos_output, [1, max_query_length], dim=0)

    assert torch.allclose(padded_xpos_output, xpos_output, atol=1e-5)

    if need_weights:
        _, padded_xpos_attn_weights = torch.split(
            padded_xpos_attn_weights, [1, max_query_length], dim=-2
        )
        _, padded_xpos_attn_weights = torch.split(
            padded_xpos_attn_weights, [1, max_key_length], dim=-1
        )

        assert torch.allclose(padded_xpos_attn_weights, xpos_attn_weights, atol=1e-6)

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        embed_dim,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_attn_mask:
        attn_mask = None

    xpos_mha = ExtrapolatablePositionalMultiheadAttention(
        embed_dim,
        num_heads,
        bias=bias,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    xpos_output, xpos_attn_weights = xpos_mha(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
        attn_mask=attn_mask,
        need_weights=need_weights,
        average_attn_weights=need_weights,
    )

    if batch_first:
        assert xpos_output.size() == (batch_size, max_query_length, embed_dim)
    else:
        assert xpos_output.size() == (max_query_length, batch_size, embed_dim)

    if need_weights:
        assert xpos_attn_weights.size() == (batch_size, max_query_length, max_key_length)


def create_qkv(
    batch_size: int,
    max_query_length: int,
    max_key_length: int,
    embed_dim: int,
    kdim: Optional[int] = None,
    vdim: Optional[int] = None,
    batch_first: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.LongTensor, torch.LongTensor]
]:
    if kdim is None:
        kdim = embed_dim

    if vdim is None:
        vdim = embed_dim

    query_length = torch.randint(max_query_length // 2, max_query_length, (batch_size,))
    max_query_length = torch.max(query_length).item()
    key_length = torch.randint(max_key_length // 2, max_key_length, (batch_size,))
    max_key_length = torch.max(key_length).item()

    query = torch.randn(max_query_length, batch_size, embed_dim)
    key = torch.randn(max_key_length, batch_size, embed_dim)
    value = torch.randn(max_key_length, batch_size, embed_dim)

    if batch_first:
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)

    return (query, key, value), (query_length, key_length)


def create_padding_masks(
    query_length: torch.LongTensor,
    key_length: torch.LongTensor,
) -> Tuple[torch.BoolTensor, torch.BoolTensor]:
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    query_pos_indices = torch.arange(max_query_length)
    key_pos_indices = torch.arange(max_key_length)
    key_padding_mask = key_pos_indices >= key_length.unsqueeze(dim=-1)

    if max_query_length > max_key_length:
        attn_mask = key_pos_indices > query_pos_indices.unsqueeze(dim=-1)
    else:
        attn_mask = key_pos_indices < query_pos_indices.unsqueeze(dim=-1)
        attn_mask = torch.flip(attn_mask, dims=(-2, -1))

    return key_padding_mask, attn_mask
