import pytest
import torch
import torch.nn as nn

from audyn.modules.activation import (
    AbsolutePositionalMultiheadSelfAttention,
    RelativePositionalMultiheadSelfAttention,
)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("share_heads", [True, False])
def test_absolute_positional_self_attn(
    batch_first: bool, is_causal: bool, share_heads: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_pos_length, max_length, embed_dim = 16, 12, 8
    num_heads = 4

    length = torch.randint(max_length // 2, max_length, (batch_size,))
    max_length = torch.max(length).item()

    input = torch.randn(max_length, batch_size, embed_dim)

    if batch_first:
        input = input.transpose(1, 0)

    pos_indices = torch.arange(max_length)
    padding_mask = pos_indices >= length.unsqueeze(dim=-1)

    if is_causal:
        attn_mask = pos_indices > pos_indices.unsqueeze(dim=-1)
    else:
        attn_mask = None

    absolute_mha = AbsolutePositionalMultiheadSelfAttention(
        embed_dim,
        num_heads,
        max_length=max_pos_length,
        share_heads=share_heads,
        batch_first=batch_first,
    )

    absolute_output, absolute_attn_weights = absolute_mha(
        input,
        padding_mask=padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert absolute_output.size() == (batch_size, max_length, embed_dim)
    else:
        assert absolute_output.size() == (max_length, batch_size, embed_dim)

    assert absolute_attn_weights.size() == (batch_size, max_length, max_length)

    # compatibility with nn.MultiheadAttention
    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        batch_first=batch_first,
    )

    absolute_mha.q_pos_emb.data.zero_()
    absolute_mha.k_pos_emb.data.zero_()
    absolute_mha.v_pos_emb.data.zero_()

    absolute_mha.in_proj_weight.data.copy_(mha.in_proj_weight.data)
    absolute_mha.in_proj_bias.data.copy_(mha.in_proj_bias.data)
    absolute_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)
    absolute_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)

    output, attn_weights = mha(
        input,
        input,
        input,
        key_padding_mask=padding_mask,
        attn_mask=attn_mask,
    )
    absolute_output, absolute_attn_weights = absolute_mha(
        input,
        padding_mask=padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert absolute_output.size() == (batch_size, max_length, embed_dim)
    else:
        assert absolute_output.size() == (max_length, batch_size, embed_dim)

    assert absolute_attn_weights.size() == (batch_size, max_length, max_length)

    assert torch.allclose(output, absolute_output, atol=1e-7)
    assert torch.allclose(attn_weights, absolute_attn_weights)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("longer_window", [True, False])
def test_relative_positional_self_attn(
    batch_first: bool, is_causal: bool, longer_window: bool
) -> None:
    torch.manual_seed(0)

    batch_size = 3
    max_length, embed_dim = 12, 8
    num_heads = 4

    length = torch.randint(max_length // 2, max_length, (batch_size,))
    max_length = torch.max(length).item()

    if longer_window:
        window_size = max_length
    else:
        window_size = max_length - 2

    input = torch.randn(max_length, batch_size, embed_dim)

    if batch_first:
        input = input.transpose(1, 0)

    pos_indices = torch.arange(max_length)
    padding_mask = pos_indices >= length.unsqueeze(dim=-1)

    if is_causal:
        attn_mask = pos_indices > pos_indices.unsqueeze(dim=-1)
    else:
        attn_mask = None

    relative_mha = RelativePositionalMultiheadSelfAttention(
        embed_dim,
        num_heads,
        window_size=window_size,
        batch_first=batch_first,
    )

    relative_output, relative_attn_weights = relative_mha(
        input,
        padding_mask=padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_length, embed_dim)
    else:
        assert relative_output.size() == (max_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_length, max_length)

    # compatibility with nn.MultiheadAttention
    mha = nn.MultiheadAttention(
        embed_dim,
        num_heads,
        batch_first=batch_first,
    )

    relative_mha.k_pos_emb.data.zero_()
    relative_mha.v_pos_emb.data.zero_()

    relative_mha.in_proj_weight.data.copy_(mha.in_proj_weight.data)
    relative_mha.in_proj_bias.data.copy_(mha.in_proj_bias.data)
    relative_mha.out_proj.weight.data.copy_(mha.out_proj.weight.data)
    relative_mha.out_proj.bias.data.copy_(mha.out_proj.bias.data)

    output, attn_weights = mha(
        input,
        input,
        input,
        key_padding_mask=padding_mask,
        attn_mask=attn_mask,
    )
    relative_output, relative_attn_weights = relative_mha(
        input,
        padding_mask=padding_mask,
        attn_mask=attn_mask,
    )

    if batch_first:
        assert relative_output.size() == (batch_size, max_length, embed_dim)
    else:
        assert relative_output.size() == (max_length, batch_size, embed_dim)

    assert relative_attn_weights.size() == (batch_size, max_length, max_length)

    assert torch.allclose(output, relative_output, atol=1e-7)
    assert torch.allclose(attn_weights, relative_attn_weights)
