import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from ..functional.activation import scaled_dot_product_attention
from .positional_encoding import ExtrapolatablePositionalEmbedding, RotaryPositionalEmbedding

__all__ = [
    "TrainableAbsolutePositionalMultiheadAttention",
    "RelativePositionalMultiheadAttention",
    "RotaryPositionalMultiheadAttention",
    "ExtrapolatablePositionalMultiheadAttention",
]

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")


class _MultiheadAttention(nn.MultiheadAttention):
    """Wrapper class of nn.MultiheadAttention."""

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
            dropout,
            bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.validate_kwargs(kwargs)

        return super().forward(
            query,
            key,
            value,
            key_padding_mask=padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            **kwargs,
        )

    def validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate keyword arguments for backward compatibility."""
        valid_keys = set()

        if not IS_TORCH_LT_2_0:
            valid_keys.add("is_causal")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."


class TrainableAbsolutePositionalMultiheadAttention(_MultiheadAttention):
    """Multihead attention using trainable absolute positional representation."""

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
        max_length: int = 512,
        share_heads: bool = True,
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

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        if share_heads:
            embedding_shape = (max_length, embed_dim // num_heads)
        else:
            embedding_shape = (max_length, embed_dim)

        self.q_positional_embedding = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.k_positional_embedding = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.v_positional_embedding = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )

        self.max_length = max_length
        self.share_heads = share_heads

        self._reset_embeddings()

    def _reset_parameters(self) -> None:
        """Reset parameters.

        Since this method is often called before defining positional encodings,
        we initialize them only if they exist.

        """
        super()._reset_parameters()

        if (
            hasattr(self, "q_positional_embedding")
            and hasattr(self, "k_positional_embedding")
            and hasattr(self, "v_positional_embedding")
        ):
            self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        head_dim = self.embed_dim // self.num_heads

        std = math.sqrt(head_dim)
        self.q_positional_embedding.data.normal_(std=1 / std)
        self.k_positional_embedding.data.normal_(std=1 / std)
        self.v_positional_embedding.data.normal_(std=1 / std)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of AbsolutePositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        max_length = self.max_length
        q_positional_embedding = self.q_positional_embedding
        k_positional_embedding = self.k_positional_embedding
        v_positional_embedding = self.v_positional_embedding

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

        assert (
            query_length <= max_length
        ), f"Query length should be smaller than or equal to {max_length}."
        assert (
            key_length <= max_length
        ), f"Key length should be smaller than or equal to {max_length}."

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

        q = self._apply_positional_embedding(q, q_positional_embedding)
        k = self._apply_positional_embedding(k, k_positional_embedding)
        v = self._apply_positional_embedding(v, v_positional_embedding)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

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

    def _apply_positional_embedding(
        self, input: torch.Tensor, positional_embedding: torch.Tensor
    ) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            input (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as input.

        """
        max_length = self.max_length
        share_heads = self.share_heads
        input_length, _, num_heads, head_dim = input.size()

        positional_embedding, _ = torch.split(
            positional_embedding, [input_length, max_length - input_length], dim=0
        )

        if share_heads:
            positional_embedding = positional_embedding.view(input_length, 1, 1, head_dim)
        else:
            positional_embedding = positional_embedding.view(input_length, 1, num_heads, head_dim)

        output = input + positional_embedding

        return output


class RelativePositionalMultiheadAttention(_MultiheadAttention):
    """Multihead attention using relative positional representation."""

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
        window_size: int = None,
        share_heads: bool = True,
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

        if share_heads:
            embedding_shape = (2 * window_size + 1, embed_dim // num_heads)
        else:
            embedding_shape = (2 * window_size + 1, embed_dim)

        self.k_positional_embedding = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.v_positional_embedding = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )

        self.window_size = window_size
        self.share_heads = share_heads

        self._reset_embeddings()

    def _reset_parameters(self) -> None:
        """Reset parameters.

        Since this method is often called before defining positional encodings,
        we initialize them only if they exist.

        """
        super()._reset_parameters()

        if hasattr(self, "k_positional_embedding") and hasattr(self, "v_positional_embedding"):
            self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        head_dim = self.embed_dim // self.num_heads

        std = math.sqrt(head_dim)
        self.k_positional_embedding.data.normal_(std=1 / std)
        self.v_positional_embedding.data.normal_(std=1 / std)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RelativePositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        share_heads = self.share_heads
        k_positional_embedding = self.k_positional_embedding
        v_positional_embedding = self.v_positional_embedding

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

        # offset for key
        if share_heads:
            k_positional_embedding = self.expand_embedding(
                k_positional_embedding,
                query_length=query_length,
                key_length=key_length,
                num_heads=1,
            )
        else:
            k_positional_embedding = self.expand_embedding(
                k_positional_embedding,
                query_length=query_length,
                key_length=key_length,
                num_heads=num_heads,
            )

        q = q.permute(1, 2, 0, 3)
        k_positional_embedding = k_positional_embedding.permute(0, 2, 1, 3)
        k_offset = torch.matmul(q, k_positional_embedding) / math.sqrt(head_dim)
        k_offset = k_offset.permute(2, 0, 1, 3)
        qk = qk + k_offset

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

        # offset for value
        if share_heads:
            v_positional_embedding = self.expand_embedding(
                v_positional_embedding,
                query_length=query_length,
                key_length=key_length,
                num_heads=1,
            )
        else:
            v_positional_embedding = self.expand_embedding(
                v_positional_embedding,
                query_length=query_length,
                key_length=key_length,
                num_heads=num_heads,
            )

        _attn_weights = attn_weights.permute(1, 2, 0, 3)
        v_positional_embedding = v_positional_embedding.permute(0, 2, 3, 1)
        v_offset = torch.matmul(_attn_weights, v_positional_embedding)
        v_offset = v_offset.permute(2, 0, 1, 3)
        qkv = qkv + v_offset

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
        query_length: int,
        key_length: int,
        num_heads: Optional[int] = None,
    ) -> torch.Tensor:
        """Expand embedding.

        Args:
            positional_embedding (torch.Tensor): Positional embedding
                of shape (2 * window_size + 1, embed_dim).
            query_length (int): Query sequence length.
            key_length (int): Key sequence length.
            num_heads (int, optional): Number of heads.

        Returns:
            torch.Tensor: Expanded relative positional embedding
                of shape (num_heads, embed_dim // num_heads, query_length, key_length)
                if num_heads is specified. Otherwise, shape is
                (embed_dim, query_length, key_length).

        """
        window_size, embed_dim = positional_embedding.size()
        half_window_size = (window_size - 1) // 2
        left_padding = query_length - half_window_size - 1
        right_padding = key_length - half_window_size - 1

        left_positional_embedding, center_positional_embedding, right_positional_embedding = (
            torch.split(positional_embedding, [half_window_size, 1, half_window_size], dim=0)
        )

        if left_padding > 0:
            left_positional_embedding, middle_positional_embedding = torch.split(
                left_positional_embedding, [1, half_window_size - 1], dim=0
            )
            left_positional_embedding = left_positional_embedding.expand(
                (left_padding + 1, embed_dim)
            )
            left_positional_embedding = torch.cat(
                [left_positional_embedding, middle_positional_embedding], dim=0
            )
        else:
            left_positional_embedding = F.pad(left_positional_embedding, (0, 0, left_padding, 0))

        if right_padding > 0:
            middle_positional_embedding, right_positional_embedding = torch.split(
                right_positional_embedding, [half_window_size - 1, 1], dim=0
            )
            right_positional_embedding = right_positional_embedding.expand(
                (right_padding + 1, embed_dim)
            )
            right_positional_embedding = torch.cat(
                [middle_positional_embedding, right_positional_embedding], dim=0
            )
        else:
            right_positional_embedding = F.pad(
                right_positional_embedding, (0, 0, 0, right_padding)
            )

        positional_embedding = torch.cat(
            [left_positional_embedding, center_positional_embedding, right_positional_embedding],
            dim=0,
        )
        positional_embedding = positional_embedding.view(
            1, 1, query_length + key_length - 1, embed_dim
        )
        positional_embedding = positional_embedding.permute(0, 3, 1, 2)
        positional_embedding = F.unfold(
            positional_embedding, kernel_size=(1, key_length), stride=1
        )
        positional_embedding = positional_embedding.view(embed_dim, key_length, query_length)
        positional_embedding = positional_embedding.permute(0, 2, 1)
        positional_embedding = torch.flip(positional_embedding, dims=(-2,))

        if num_heads is not None:
            positional_embedding = positional_embedding.view(
                num_heads, embed_dim // num_heads, query_length, key_length
            )

        return positional_embedding


class RotaryPositionalMultiheadAttention(_MultiheadAttention):
    """Multihead attention using rotary positional representation."""

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
        base: int = 10000,
        share_heads: bool = True,
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

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self.rope = RotaryPositionalEmbedding(base=base, batch_first=batch_first)

        self.share_heads = share_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RotaryPositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

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

        q = self._apply_positional_embedding(q.contiguous())
        k = self._apply_positional_embedding(k.contiguous())

        q = q.view(query_length, batch_size, num_heads, head_dim)
        k = k.view(key_length, batch_size, num_heads, head_dim)
        v = v.view(key_length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

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

    def _apply_positional_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            input (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as input.

        """
        share_heads = self.share_heads
        num_heads = self.num_heads
        input_length, batch_size, embed_dim = input.size()
        head_dim = embed_dim // num_heads

        if share_heads:
            x = input.view(input_length, batch_size * num_heads, head_dim)
        else:
            x = input.view(input_length, batch_size, num_heads * head_dim)

        if self.batch_first:
            x = x.transpose(1, 0)

        x = self.rope(x)

        if self.batch_first:
            x = x.transpose(1, 0).contiguous()

        output = x.view(input_length, batch_size, embed_dim)

        return output


class ExtrapolatablePositionalMultiheadAttention(_MultiheadAttention):
    """Multihead attention using extrapolatable positional representation."""

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
        base: int = 10000,
        share_heads: bool = True,
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

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self.q_xpos = ExtrapolatablePositionalEmbedding(False, base=base, batch_first=batch_first)
        self.k_xpos = ExtrapolatablePositionalEmbedding(True, base=base, batch_first=batch_first)

        self.share_heads = share_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of ExtrapolatablePositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

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

        q = self._apply_q_positional_embedding(q.contiguous())
        k = self._apply_k_positional_embedding(k.contiguous())

        q = q.view(query_length, batch_size, num_heads, head_dim)
        k = k.view(key_length, batch_size, num_heads, head_dim)
        v = v.view(key_length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

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

    def _apply_q_positional_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to query.

        Args:
            query (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as query.

        """
        output = self._apply_positional_embedding(input, xpos=self.q_xpos)

        return output

    def _apply_k_positional_embedding(self, input: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to key.

        Args:
            key (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as key.

        """
        output = self._apply_positional_embedding(input, xpos=self.k_xpos)

        return output

    def _apply_positional_embedding(
        self, input: torch.Tensor, xpos: ExtrapolatablePositionalEmbedding
    ) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            input (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).
            xpos (ExtrapolatablePositionalEmbedding): xPos for query or key.

        Returns:
            torch.Tensor: Output sequence same shape as input.

        """
        share_heads = self.share_heads
        num_heads = self.num_heads
        input_length, batch_size, embed_dim = input.size()
        head_dim = embed_dim // num_heads

        if share_heads:
            x = input.view(input_length, batch_size * num_heads, head_dim)
        else:
            x = input.view(input_length, batch_size, num_heads * head_dim)

        if self.batch_first:
            x = x.transpose(1, 0)

        x = xpos(x)

        if self.batch_first:
            x = x.transpose(1, 0).contiguous()

        output = x.view(input_length, batch_size, embed_dim)

        return output
