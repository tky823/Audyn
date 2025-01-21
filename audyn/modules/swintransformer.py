import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from ..modules.activation import _MultiheadAttention

__all__ = [
    "SwinRelativePositionalMultiheadAttention",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class SwinRelativePositionalMultiheadAttention(_MultiheadAttention):
    """Multihead attention using relative positional representation for SwinTransformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        window_size: _size_2_t = None,
        share_heads: bool = False,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )

        if window_size is None:
            raise ValueError("Specify window size.")

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        window_size = _pair(window_size)
        window_height, window_width = window_size

        if share_heads:
            embedding_shape = (2 * window_height - 1, 2 * window_width - 1, 1)
        else:
            embedding_shape = (2 * window_height - 1, 2 * window_width - 1, num_heads)

        self.positional_embedding = nn.Parameter(
            torch.zeros(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )

        self.window_size = window_size
        self.share_heads = share_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        need_weights: bool = True,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of SwinRelativePositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape
                (batch_size, window_height * window_width, embed_dim) if ``batch_first=True``,
                otherwise (window_height * window_width, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape
                (batch_size, window_height * window_width, embed_dim) if ``batch_first=True``,
                otherwise (window_height * window_width, batch_size, embed_dim).
            value (torch.Tensor): Sequence of shape
                (batch_size, window_height * window_width, embed_dim) if ``batch_first=True``,
                otherwise (window_height * window_width, batch_size, embed_dim).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, window_height * window_width, window_height * window_width)
                    if ``average_attn_weights=True``, otherwise
                    (batch_size, window_height * window_width, window_height * window_width).

        """  # noqa: E501
        key_padding_mask = None
        attn_mask = None

        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        share_heads = self.share_heads
        positional_embedding = self.positional_embedding

        head_dim = embed_dim // num_heads

        if batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query = query.transpose(1, 0)
                    key = key.transpose(1, 0)
                    value = key
            else:
                query = query.transpose(1, 0)
                key = key.transpose(1, 0)
                value = value.transpose(1, 0)

        query_length, batch_size, _ = query.size()
        key_length, _, _ = key.size()

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

        if self._qkv_same_embed_dim:
            q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
                in_proj_weight, [embed_dim] * 3, dim=-2
            )
        else:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight

        if self.in_proj_bias is None:
            q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = F.linear(query, q_proj_weight, bias=q_proj_bias)
        k = F.linear(key, k_proj_weight, bias=k_proj_bias)
        v = F.linear(value, v_proj_weight, bias=v_proj_bias)

        q = q.view(query_length, batch_size, num_heads, head_dim)
        k = k.view(key_length, batch_size, num_heads, head_dim)
        v = v.view(key_length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 3, 0)
        v = v.permute(1, 2, 0, 3)
        qk = torch.matmul(q, k) / math.sqrt(head_dim)

        if share_heads:
            offset = self.expand_embedding(
                positional_embedding,
                num_heads=1,
            )
        else:
            offset = self.expand_embedding(
                positional_embedding,
                num_heads=num_heads,
            )

        qk = qk + offset

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, key_length)

            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                if attn_mask.dim() == 3:
                    attn_mask.view(batch_size, num_heads, query_length, key_length)
                else:
                    assert attn_mask.dim() == 2

                attn_mask = attn_mask + key_padding_mask

        if attn_mask is not None:
            qk = qk + attn_mask

        attn_weights = F.softmax(qk, dim=-1)

        if dropout > 0:
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

        qkv = torch.matmul(attn_weights, v)

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, query_length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(query_length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights and need_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights

    @staticmethod
    def expand_embedding(
        positional_embedding: torch.Tensor,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """Expand embedding.

        Args:
            positional_embedding (torch.Tensor): Positional embedding
                of shape (2 * window_height - 1, 2 * window_width - 1, 1)
                or (2 * window_height - 1, 2 * window_width - 1, num_heads).
            num_heads (int, optional): Number of heads.

        Returns:
            torch.Tensor: Expanded relative positional embedding
                of shape (num_heads, window_height * window_width, window_height * window_width)
                if num_heads is specified. Otherwise, shape is
                (window_height * window_width, window_height * window_width).

        """  # noqa: E501
        long_window_height, long_window_width, _ = positional_embedding.size()
        window_height = (long_window_height + 1) // 2
        window_width = (long_window_width + 1) // 2

        positional_embedding = positional_embedding.permute(2, 0, 1).contiguous()
        positional_embedding = positional_embedding.unsqueeze(dim=0)
        positional_embedding = F.unfold(
            positional_embedding, kernel_size=(window_height, window_width), stride=(1, 1)
        )
        positional_embedding = positional_embedding.view(
            -1, window_height * window_width, window_height, window_width
        )
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)
        positional_embedding = torch.flip(positional_embedding, dims=(1, 2))
        positional_embedding = positional_embedding.view(
            -1, window_height * window_width, window_height * window_width
        )

        if num_heads is None:
            positional_embedding = positional_embedding.view(
                window_height * window_width,
                window_height * window_width,
            )
        else:
            positional_embedding = positional_embedding.view(
                num_heads,
                window_height * window_width,
                window_height * window_width,
            )

        return positional_embedding
