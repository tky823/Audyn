import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from .positional_encoding import RotaryPositionalEmbedding

__all__ = [
    "MultiheadSelfAttention",
    "AbsolutePositionalMultiheadSelfAttention",
    "RelativePositionalMultiheadSelfAttention",
    "RotaryPositionalMultiheadSelfAttention",
]


class MultiheadSelfAttention(nn.MultiheadAttention):
    """Wrapper class of nn.MultiheadAttention for self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        # Key and value are identical to query.
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=None,
            vdim=None,
            batch_first=batch_first,
            **factory_kwargs,
        )

        if not self._qkv_same_embed_dim:
            raise ValueError(
                "Embedding dimensions of key and value should be equal to "
                "that of query in self-attention."
            )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        valid_keys = set()

        if version.parse(torch.__version__) < version.parse("2.0"):
            valid_keys.add("is_causal")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."

        return super().forward(
            input,
            input,
            input,
            key_padding_mask=padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            **kwargs,
        )


class AbsolutePositionalMultiheadSelfAttention(MultiheadSelfAttention):
    """Multihead self-attention using trainable absolute positional representation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
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

        self.q_pos_emb = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.k_pos_emb = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.v_pos_emb = nn.Parameter(
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
            hasattr(self, "q_pos_emb")
            and hasattr(self, "k_pos_emb")
            and hasattr(self, "v_pos_emb")
        ):
            self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        head_dim = self.embed_dim // self.num_heads

        std = math.sqrt(head_dim)
        self.q_pos_emb.data.normal_(std=1 / std)
        self.k_pos_emb.data.normal_(std=1 / std)
        self.v_pos_emb.data.normal_(std=1 / std)

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of AbsolutePositionalMultiheadSelfAttention.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embed_dim)
                if ``batch_first=True``, otherwise (length, batch_size, embed_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (length, length) or (batch_size * num_heads, length, length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape (batch_size, num_heads, length, length)
                    if ``average_attn_weights=True``, otherwise (batch_size, length, length).

        """
        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        max_length = self.max_length
        q_pos_emb = self.q_pos_emb
        k_pos_emb = self.k_pos_emb
        v_pos_emb = self.v_pos_emb

        head_dim = embed_dim // num_heads

        if batch_first:
            x = input.transpose(1, 0)
        else:
            x = input

        length, batch_size, _ = x.size()

        assert (
            length <= max_length
        ), f"Sequence length should be smaller than or equal to {max_length}."

        padding_mask = F._canonical_mask(
            mask=padding_mask,
            mask_name="padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=x.dtype,
            check_other=False,
        )

        x = F.linear(x, in_proj_weight, bias=in_proj_bias)

        q, k, v = torch.chunk(x, chunks=3, dim=-1)
        q = q.view(length, batch_size, num_heads, head_dim)
        k = k.view(length, batch_size, num_heads, head_dim)
        v = v.view(length, batch_size, num_heads, head_dim)

        q = self._apply_pos_emb(q, q_pos_emb)
        k = self._apply_pos_emb(k, k_pos_emb)
        v = self._apply_pos_emb(v, v_pos_emb)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 3, 0)
        v = v.permute(1, 2, 0, 3)
        qk = torch.matmul(q, k) / math.sqrt(head_dim)

        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, 1, 1, length)

            if attn_mask is None:
                attn_mask = padding_mask
            else:
                if attn_mask.dim() == 3:
                    attn_mask.view(batch_size, num_heads, length, length)
                else:
                    assert attn_mask.dim() == 2

                attn_mask = attn_mask + padding_mask

        if attn_mask is not None:
            qk = qk + attn_mask

        attn_weights = F.softmax(qk, dim=-1)

        if dropout > 0:
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

        qkv = torch.matmul(attn_weights, v)

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights

    def _apply_pos_emb(self, input: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """Apply positional embedding to input.

        Args:
            input (torch.Tensor): Sequence of shape (length, batch_size, num_heads, head_dim).

        Returns:
            torch.Tensor: Output sequence same shape as input.

        """
        max_length = self.max_length
        share_heads = self.share_heads
        input_length, _, num_heads, head_dim = input.size()

        pos_emb, _ = torch.split(pos_emb, [input_length, max_length - input_length], dim=0)

        if share_heads:
            pos_emb = pos_emb.view(input_length, 1, 1, head_dim)
        else:
            pos_emb = pos_emb.view(input_length, 1, num_heads, head_dim)

        output = input + pos_emb

        return output


class RelativePositionalMultiheadSelfAttention(MultiheadSelfAttention):
    """Multihead self-attention using relative positional representation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
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

        self.k_pos_emb = nn.Parameter(
            torch.empty(
                *embedding_shape,
                **factory_kwargs,
            ),
            requires_grad=True,
        )
        self.v_pos_emb = nn.Parameter(
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

        if hasattr(self, "k_pos_emb") and hasattr(self, "v_pos_emb"):
            self._reset_embeddings()

    def _reset_embeddings(self) -> None:
        head_dim = self.embed_dim // self.num_heads

        std = math.sqrt(head_dim)
        self.k_pos_emb.data.normal_(std=1 / std)
        self.v_pos_emb.data.normal_(std=1 / std)

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RelativePositionalMultiheadSelfAttention.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embed_dim)
                if ``batch_first=True``, otherwise (length, batch_size, embed_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (length, length) or (batch_size * num_heads, length, length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape (batch_size, num_heads, length, length)
                    if ``average_attn_weights=True``, otherwise (batch_size, length, length).

        """
        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        share_heads = self.share_heads
        k_pos_emb = self.k_pos_emb
        v_pos_emb = self.v_pos_emb

        head_dim = embed_dim // num_heads

        if batch_first:
            x = input.transpose(1, 0)
        else:
            x = input

        length, batch_size, _ = x.size()

        padding_mask = F._canonical_mask(
            mask=padding_mask,
            mask_name="padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=x.dtype,
            check_other=False,
        )

        x = F.linear(x, in_proj_weight, bias=in_proj_bias)

        q, k, v = torch.chunk(x, chunks=3, dim=-1)
        q = q.view(length, batch_size, num_heads, head_dim)
        k = k.view(length, batch_size, num_heads, head_dim)
        v = v.view(length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 3, 0)
        v = v.permute(1, 2, 0, 3)
        qk = torch.matmul(q, k) / math.sqrt(head_dim)

        # offset for key
        if share_heads:
            k_pos_emb = self.expand_embedding(k_pos_emb, length=length, num_heads=1)
        else:
            k_pos_emb = self.expand_embedding(k_pos_emb, length=length, num_heads=num_heads)

        q = q.permute(1, 2, 0, 3)
        k_pos_emb = k_pos_emb.permute(0, 2, 1, 3)
        k_offset = torch.matmul(q, k_pos_emb) / math.sqrt(head_dim)
        k_offset = k_offset.permute(2, 0, 1, 3)
        qk = qk + k_offset

        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, 1, 1, length)

            if attn_mask is None:
                attn_mask = padding_mask
            else:
                if attn_mask.dim() == 3:
                    attn_mask.view(batch_size, num_heads, length, length)
                else:
                    assert attn_mask.dim() == 2

                attn_mask = attn_mask + padding_mask

        if attn_mask is not None:
            qk = qk + attn_mask

        attn_weights = F.softmax(qk, dim=-1)

        if dropout > 0:
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

        qkv = torch.matmul(attn_weights, v)

        # offset for value
        if share_heads:
            v_pos_emb = self.expand_embedding(v_pos_emb, length=length, num_heads=1)
        else:
            v_pos_emb = self.expand_embedding(v_pos_emb, length=length, num_heads=num_heads)

        _attn_weights = attn_weights.permute(1, 3, 0, 2)
        v_pos_emb = v_pos_emb.permute(0, 3, 2, 1)
        v_offset = torch.matmul(_attn_weights, v_pos_emb)
        v_offset = v_offset.permute(2, 0, 1, 3)
        qkv = qkv + v_offset

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights

    @staticmethod
    def expand_embedding(
        pos_embedding: torch.Tensor, length: int, num_heads: Optional[int] = None
    ) -> torch.Tensor:
        """Expand embedding.

        Args:
            pos_embedding (torch.Tensor): Positional embedding
                of shape (2 * window_size + 1, embed_dim).
            length (int): Sequence length.
            num_heads (int, optional): Number of heads.

        Returns:
            torch.Tensor: Expanded relative positional embedding
                of shape (num_heads, embed_dim // num_heads, length, length)
                if num_heads is specified. Otherwise, shape is
                (embed_dim, length, length).

        """
        window_size, embed_dim = pos_embedding.size()
        window_size = (window_size - 1) // 2
        padding = window_size - length + 1

        if padding < 0:
            left_pos_embedding, center_pos_embedding, right_pos_embedding = torch.split(
                pos_embedding, [1, 2 * window_size - 1, 1], dim=0
            )
            left_pos_embedding = left_pos_embedding.expand((length - window_size, embed_dim))
            right_pos_embedding = right_pos_embedding.expand((length - window_size, embed_dim))
            pos_embedding = torch.cat(
                [left_pos_embedding, center_pos_embedding, right_pos_embedding], dim=0
            )
        else:
            window_size < length - 1
            pos_embedding = F.pad(pos_embedding, (0, 0, -padding, -padding))

        pos_embedding = pos_embedding.view(1, 1, 2 * length - 1, embed_dim)
        pos_embedding = pos_embedding.permute(0, 3, 1, 2)
        pos_embedding = F.unfold(pos_embedding, kernel_size=(1, length), stride=1)

        if num_heads is None:
            pos_embedding = pos_embedding.view(embed_dim, length, -1)
        else:
            pos_embedding = pos_embedding.view(num_heads, embed_dim // num_heads, length, -1)

        pos_embedding = torch.flip(pos_embedding, dims=(-2,))

        return pos_embedding


class RotaryPositionalMultiheadSelfAttention(MultiheadSelfAttention):
    """Multihead self-attention using rotary positional representation."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        base: int = 10000,
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
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RotaryPositionalMultiheadSelfAttention.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embed_dim)
                if ``batch_first=True``, otherwise (length, batch_size, embed_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape (batch_size, length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (length, length) or (batch_size * num_heads, length, length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape (batch_size, num_heads, length, length)
                    if ``average_attn_weights=True``, otherwise (batch_size, length, length).

        """
        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

        head_dim = embed_dim // num_heads

        if batch_first:
            x = input.transpose(1, 0)
        else:
            x = input

        length, batch_size, _ = x.size()

        padding_mask = F._canonical_mask(
            mask=padding_mask,
            mask_name="padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=x.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=x.dtype,
            check_other=False,
        )

        x = F.linear(x, in_proj_weight, bias=in_proj_bias)

        q, k, v = torch.chunk(x, chunks=3, dim=-1)

        q = self._apply_pos_emb(q.contiguous())
        k = self._apply_pos_emb(k.contiguous())

        q = q.view(length, batch_size, num_heads, head_dim)
        k = k.view(length, batch_size, num_heads, head_dim)
        v = v.view(length, batch_size, num_heads, head_dim)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 3, 0)
        v = v.permute(1, 2, 0, 3)
        qk = torch.matmul(q, k) / math.sqrt(head_dim)

        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, 1, 1, length)

            if attn_mask is None:
                attn_mask = padding_mask
            else:
                if attn_mask.dim() == 3:
                    attn_mask.view(batch_size, num_heads, length, length)
                else:
                    assert attn_mask.dim() == 2

                attn_mask = attn_mask + padding_mask

        if attn_mask is not None:
            qk = qk + attn_mask

        attn_weights = F.softmax(qk, dim=-1)

        if dropout > 0:
            attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)

        qkv = torch.matmul(attn_weights, v)

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights

    def _apply_pos_emb(self, input: torch.Tensor) -> torch.Tensor:
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
