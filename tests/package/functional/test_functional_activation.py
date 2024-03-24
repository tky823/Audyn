import math
from typing import Tuple

import pytest
import torch
import torch.nn.functional as F
from dummy import allclose
from packaging import version

from audyn.functional.activation import scaled_dot_product_attention


@pytest.mark.parametrize("use_key_padding_mask", [True, False])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.skipif(
    version.parse(torch.__version__) < version.parse("2.0"), reason="torch >= 2.0 is required."
)
def test_scaled_dot_product_attention(use_key_padding_mask: bool, use_attn_mask: bool) -> None:
    torch.manual_seed(0)

    batch_first = True
    num_heads, head_dim = 8, 6
    max_query_length, max_key_length = 12, 10
    batch_size, query_length, key_length = 4, 10, 12

    (query, key, value), (query_length, key_length) = create_qkv(
        batch_size,
        max_query_length,
        max_key_length,
        num_heads,
        head_dim,
        batch_first=batch_first,
    )
    max_query_length = torch.max(query_length).item()
    max_key_length = torch.max(key_length).item()

    if batch_first:
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
    else:
        query = query.transpose(0, 2)
        key = key.transpose(0, 2)
        value = value.transpose(0, 2)

    key_padding_mask, attn_mask = create_padding_masks(query_length, key_length)

    if not use_key_padding_mask:
        key_padding_mask = None

    if not use_attn_mask:
        attn_mask = None

    output_sdpa, _ = scaled_dot_product_attention(
        query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
    )

    key_padding_mask = F._canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=F._none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype,
    )

    attn_mask = F._canonical_mask(
        mask=attn_mask,
        mask_name="attn_mask",
        other_type=None,
        other_name="",
        target_type=query.dtype,
        check_other=False,
    )

    if key_padding_mask is not None:
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, max_key_length)

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            if attn_mask.dim() == 3:
                attn_mask.view(batch_size, num_heads, -1, max_key_length)
            else:
                assert attn_mask.dim() == 2

            attn_mask = attn_mask + key_padding_mask

    key = key.transpose(-2, -1)
    qk = torch.matmul(query, key) / math.sqrt(head_dim)

    if attn_mask is not None:
        qk = qk + attn_mask

    attn_weights = F.softmax(qk, dim=-1)

    output_naive = torch.matmul(attn_weights, value)

    allclose(output_sdpa, output_naive, atol=1e-6)


def create_qkv(
    batch_size: int,
    max_query_length: int,
    max_key_length: int,
    num_heads: int,
    head_dim: int,
    batch_first: bool = False,
) -> Tuple[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.LongTensor, torch.LongTensor]
]:
    query_length = torch.randint(max_query_length // 2, max_query_length, (batch_size,))
    max_query_length = torch.max(query_length).item()
    key_length = torch.randint(max_key_length // 2, max_key_length, (batch_size,))
    max_key_length = torch.max(key_length).item()

    query = torch.randn(max_query_length, batch_size, num_heads, head_dim)
    key = torch.randn(max_key_length, batch_size, num_heads, head_dim)
    value = torch.randn(max_key_length, batch_size, num_heads, head_dim)

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
