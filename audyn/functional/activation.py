import math
from typing import Optional, Tuple

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
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Wrapper function of torch.nn.functional.scaled_dot_product_attention.

    torch.nn.functional.scaled_dot_product_attention supports memory-efficient attention
    mechanisms, but it is not implemented in the previous torch (< 2.0). To absorb this difference,
    this function is designed as a compatibility layer.

    Args:
        query (torch.Tensor): Query of shape (batch_size, num_heads, query_length, head_dim).
        key (torch.Tensor): Key of shape (batch_size, num_heads, key_length, head_dim).
        value (torch.Tensor): Value of shape (batch_size, num_heads, key_length, head_dim).
        key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
            (batch_size, key_length). If ``True``, the corresponding key position
            will be ignored in the attention computation.
        attn_mask (torch.BoolTensor, optional): Attention mask of shape
            (query_length, key_length) or (batch_size * num_heads, query_length, key_length).
            If ``True``, the corresponding attention position will be ignored.
        dropout_p (float): Dropout probability. To deactivate, explicitly set ``dropout_p=0``.
            Default: 0.0.
        need_weights (bool): If ``True``, attention weights are returned. Note that this
            disables the use of memory-efficient attention and may impact performance.
            Default: ``False``.
        **kwargs: Additional keyword arguments:
            - is_causal (bool, optional): If ``True``, applies a causal mask. Supported by
                torch>=2.0.
            - scale (float, optional): Scaling factor for attention scores. Supported by
                torch>=2.1.

    Returns:
        tuple: Tuple of tensors containing:
            - torch.Tensor: Output tensor of shape (batch_size, num_heads, query_length, head_dim).
            - torch.Tensor or None: Attention weights of shape
              (batch_size, num_heads, query_length, key_length) if ``need_weights=True``,
              otherwise ``None``.

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

    # Masks are converted to float for F.scaled_dot_product_attention
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

    if key_padding_mask is None:
        if attn_mask is not None and attn_mask.dim() == 3:
            attn_mask = attn_mask.view(batch_size, num_heads, -1, key_length)
    else:
        key_padding_mask = key_padding_mask.view(batch_size, 1, 1, key_length)

        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            if attn_mask.dim() == 3:
                attn_mask = attn_mask.view(batch_size, num_heads, -1, key_length)
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


def sliding_window_multihead_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: torch.Tensor | None,
    in_proj_bias: torch.Tensor | None,
    bias_k: torch.Tensor | None,
    bias_v: torch.Tensor | None,
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    window_size: int = None,
    training: bool = True,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = True,
    attn_mask: torch.Tensor | None = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: torch.Tensor | None = None,
    k_proj_weight: torch.Tensor | None = None,
    v_proj_weight: torch.Tensor | None = None,
    average_attn_weights: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sliding window multi-head attention mechanism.

    This function implements multi-head attention with a sliding window constraint,
    where each query position only attends to a local window of key positions.
    This is useful for processing long sequences with linear memory complexity.

    Args:
        query (torch.Tensor): Query tensor of shape (query_length, batch_size, embed_dim).
        key (torch.Tensor): Key tensor of shape (key_length, batch_size, embed_dim).
        value (torch.Tensor): Value tensor of shape (key_length, batch_size, embed_dim).
        embed_dim_to_check (int): Total dimension of the model.
        num_heads (int): Number of parallel attention heads.
        in_proj_weight (torch.Tensor or None): Combined input projection weight for query,
            key, and value. Used when ``use_separate_proj_weight=False``.
        in_proj_bias (torch.Tensor or None): Combined input projection bias.
        bias_k (torch.Tensor or None): Bias for key sequence. Not supported.
        bias_v (torch.Tensor or None): Bias for value sequence. Not supported.
        add_zero_attn (bool): Whether to add a batch of zeros to key and value sequences.
            Not supported (raises NotImplementedError).
        dropout_p (float): Dropout probability applied to attention weights.
        out_proj_weight (torch.Tensor): Output projection weight.
        out_proj_bias (torch.Tensor or None): Output projection bias.
        window_size (int): Size of the sliding window. Each query position attends to positions
            within ``[position - window_size, position + window_size]``. Must be specified.
        training (bool): If ``True``, applies dropout during training. Default: ``True``.
        key_padding_mask (torch.Tensor or None): Binary mask of shape (batch_size, key_length).
            If ``True``, the corresponding key position is ignored.
        need_weights (bool): Whether to return attention weights. Not supported.
            Default: ``True``.
        attn_mask (torch.Tensor or None): Attention mask. Not supported.
        use_separate_proj_weight (bool): If ``True``, uses separate projection weights for query,
            key, and value. If ``False``, uses combined ``in_proj_weight``. Default: ``False``.
        q_proj_weight (torch.Tensor or None): Query projection weight. Required when
            ``use_separate_proj_weight=True``.
        k_proj_weight (torch.Tensor or None): Key projection weight. Required when
            ``use_separate_proj_weight=True``.
        v_proj_weight (torch.Tensor or None): Value projection weight. Required when
            ``use_separate_proj_weight=True``.
        average_attn_weights (bool): Whether to average attention weights across heads.
            Only has effect when ``need_weights=True``. Default: ``True``.

    Returns:
        tuple: Tuple of tensors containing:
            - torch.Tensor: Output tensor of shape (query_length, batch_size, embed_dim).
            - None: Attention weights are not returned (always ``None``).

    Raises:
        NotImplementedError: If ``bias_k``, ``bias_v``, or ``add_zero_attn`` is provided.
        ValueError: If ``window_size`` is not specified, or if ``attn_mask`` or
            ``need_weights`` is set to incompatible values.

    """
    if bias_k is not None:
        raise NotImplementedError("bias_k is not supported.")

    if bias_v is not None:
        raise NotImplementedError("bias_v is not supported.")

    if add_zero_attn:
        raise NotImplementedError("add_zero_attn is not supported.")

    if window_size is None:
        raise ValueError("window_size must be specified.")

    if attn_mask is not None:
        raise ValueError("attn_mask is not supported.")

    head_dim = embed_dim_to_check // num_heads

    query_length, batch_size, _ = query.size()
    key_length, _, _ = key.size()

    if use_separate_proj_weight:
        assert q_proj_weight is not None, (
            "q_proj_weight is required when use_separate_proj_weight is True."
        )
        assert k_proj_weight is not None, (
            "k_proj_weight is required when use_separate_proj_weight is True."
        )
        assert v_proj_weight is not None, (
            "v_proj_weight is required when use_separate_proj_weight is True."
        )
    else:
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            in_proj_weight, [embed_dim_to_check] * 3, dim=-2
        )

    if in_proj_bias is None:
        q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
    else:
        q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
            in_proj_bias, [embed_dim_to_check] * 3, dim=0
        )

    q = F.linear(query, q_proj_weight, bias=q_proj_bias)
    k = F.linear(key, k_proj_weight, bias=k_proj_bias)
    v = F.linear(value, v_proj_weight, bias=v_proj_bias)

    q = q.view(query_length, batch_size, num_heads, head_dim)
    k = k.view(key_length, batch_size * num_heads, head_dim, 1)
    v = v.view(key_length, batch_size * num_heads, head_dim, 1)

    k = k.permute(1, 2, 3, 0)
    v = v.permute(1, 2, 3, 0)
    k = F.pad(k, (window_size, query_length - key_length + window_size))
    v = F.pad(v, (window_size, query_length - key_length + window_size))

    k = F.unfold(k, kernel_size=(1, 2 * window_size + 1), stride=(1, 1))
    v = F.unfold(v, kernel_size=(1, 2 * window_size + 1), stride=(1, 1))
    k = k.view(batch_size, num_heads, head_dim, 2 * window_size + 1, query_length)
    v = v.view(batch_size, num_heads, head_dim, 2 * window_size + 1, query_length)
    q = q.permute(1, 0, 2, 3).contiguous()
    k = k.permute(0, 4, 1, 3, 2).contiguous()
    v = v.permute(0, 4, 1, 3, 2).contiguous()
    q = q.view(batch_size * query_length, num_heads, 1, head_dim)
    k = k.view(batch_size * query_length, num_heads, 2 * window_size + 1, head_dim)
    v = v.view(batch_size * query_length, num_heads, 2 * window_size + 1, head_dim)

    if key_padding_mask is None:
        attn_mask = torch.zeros((batch_size, key_length), dtype=torch.bool, device=k.device)
        attn_mask = F.pad(
            attn_mask, (window_size, query_length - key_length + window_size), value=True
        )
    else:
        if key_padding_mask.dtype != torch.bool:
            raise ValueError("key_padding_mask dtype must be torch.bool.")

        attn_mask = F.pad(
            key_padding_mask,
            (window_size, query_length - key_length + window_size),
            value=True,
        )

    attn_mask = attn_mask.to(k.dtype)
    attn_mask = attn_mask.view(batch_size, 1, 1, query_length + 2 * window_size)
    attn_mask = F.unfold(attn_mask, kernel_size=(1, 2 * window_size + 1), stride=(1, 1))
    attn_mask = attn_mask.to(torch.bool)
    attn_mask = attn_mask.permute(0, 2, 1).contiguous()
    attn_mask = attn_mask.view(batch_size * query_length, 1, 1, 2 * window_size + 1)
    attn_mask = attn_mask.expand(-1, num_heads, 1, -1)

    dropout_p = dropout_p if training else 0

    qkv, attn_weights = scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        need_weights=need_weights,
    )
    qkv = qkv.view(batch_size, query_length, num_heads * head_dim)
    x = F.linear(qkv, out_proj_weight, bias=out_proj_bias)
    output = x.transpose(0, 1)

    if need_weights:
        attn_weights = attn_weights.view(batch_size, query_length, num_heads, 2 * window_size + 1)
        attn_weights = attn_weights.permute(0, 2, 1, 3)
        attn_weights = F.pad(attn_weights, (0, query_length))
        attn_weights = attn_weights.view(
            batch_size, num_heads, query_length * (2 * window_size + query_length + 1)
        )
        attn_weights = F.pad(attn_weights, (0, -query_length))
        attn_weights = attn_weights.view(batch_size, num_heads, query_length, -1)
        attn_weights = F.pad(
            attn_weights, (-window_size, -window_size + key_length - query_length)
        )
        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=-3)
    else:
        attn_weights = None

    return output, attn_weights
