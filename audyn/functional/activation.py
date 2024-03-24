import math
from typing import Optional

import torch
import torch.nn.functional as F
from packaging import version

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")
IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: Optional[torch.BoolTensor] = None,
    attn_mask: Optional[torch.BoolTensor] = None,
    dropout_p: float = 0.0,
    need_weights: bool = False,
    **kwargs,
) -> torch.Tensor:
    """Wrapper function of torch.nn.functional.scaled_dot_product_attention.

    torch.nn.functional.scaled_dot_product_attention supports memory-efficient attention
    mechanisms, but it is not implemented in the previous torch (< 2.0). To absorb this difference,
    this function is designed.

    Args:
        query (torch.Tensor): Query of shape (batch_size, num_heads, query_length, head_dim).
        key (torch.Tensor): Key of shape (batch_size, num_heads, key_length, head_dim).
        value (torch.Tensor): Value of shape (batch_size, num_heads, key_length, head_dim).
        key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
            (batch_size, key_length).
        attn_mask (torch.BoolTensor, optional): Attention padding mask of
            shape (query_length, key_length) or
            (batch_size * num_heads, query_length, key_length).
        dropout_p (float): Dropout rate. To deactivate, you have to explicitly set ``dropout_p=0``.
        need_weights (bool): If ``True``, attention weight is returned.
        is_causal (bool, optional): This parameter is supported by ``torch>=2.0``.
        scale (float, optional): This parameter is supported by ``torch>=2.1``.

    Returns:
        torch.Tensor: Output of shape (batch_size, num_heads, query_length, head_dim).

    """
    if IS_TORCH_LT_2_0 or need_weights:
        use_sdpa = False
    else:
        use_sdpa = True

    valid_keys = set()

    if not IS_TORCH_LT_2_0:
        valid_keys.add("is_causal")

    if not IS_TORCH_LT_2_1:
        valid_keys.add("scale")

    invalid_keys = set(kwargs.keys()) - valid_keys

    assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."

    batch_size, num_heads, key_length, head_dim = key.size()

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
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, key_length)

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            if attn_mask.dim() == 3:
                attn_mask.view(batch_size, num_heads, -1, key_length)
            else:
                assert attn_mask.dim() == 2

            attn_mask = attn_mask + key_padding_mask

    if use_sdpa:
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            **kwargs,
        )
        attn_weights = None
    else:
        key = key.transpose(-2, -1)
        qk = torch.matmul(query, key) / math.sqrt(head_dim)

        if attn_mask is not None:
            qk = qk + attn_mask

        attn_weights = F.softmax(qk, dim=-1)

        if dropout_p > 0:
            attn_weights = F.dropout(attn_weights, p=dropout_p)

        output = torch.matmul(attn_weights, value)

    return output, attn_weights
