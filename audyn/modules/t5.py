"""T5-specific Transformer building blocks."""

from typing import Optional

import torch
import torch.nn.functional as F

from ..functional.activation import scaled_dot_product_attention
from .activation import _MultiheadAttention

__all__ = ["T5RelativePositionalMultiheadAttention"]


class T5RelativePositionalMultiheadAttention(_MultiheadAttention):
    """T5 self-attention with relative position bias inside the attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        dropout: float = 0.0,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=False,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        positional_embedding: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        out_proj_weight = self.out_proj.weight
        out_proj_bias = self.out_proj.bias
        is_self_attn = query is key and key is value

        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        if not is_self_attn:
            raise ValueError(
                "T5RelativePositionalMultiheadAttention only supports self-attention."
            )

        if is_causal:
            raise ValueError("T5 encoder attention must not be causal.")

        batch_size, query_length, embed_dim = query.size()
        key_length = key.size(1)
        head_dim = embed_dim // self.num_heads

        q, k, v = F.linear(query, self.in_proj_weight).chunk(3, dim=-1)
        q = q.view(batch_size, query_length, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, key_length, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, key_length, self.num_heads, head_dim).transpose(1, 2)
        attention_dtype = q.dtype

        if positional_embedding is None:
            attention_bias = None
        else:
            positional_embedding = positional_embedding.to(dtype=attention_dtype)

            if positional_embedding.dim() == 3:
                if positional_embedding.size(0) == self.num_heads:
                    attention_bias = positional_embedding.unsqueeze(0)
                elif positional_embedding.size(0) == batch_size * self.num_heads:
                    attention_bias = positional_embedding.view(
                        batch_size, self.num_heads, query_length, key_length
                    )
                else:
                    raise ValueError(
                        "The first dimension of a 3D positional_embedding must be "
                        "num_heads or batch_size * num_heads."
                    )
            elif positional_embedding.dim() == 4:
                attention_bias = positional_embedding
            else:
                raise ValueError("positional_embedding must have 3 or 4 dimensions.")

        padding_mask = F._canonical_mask(
            mask=padding_mask,
            mask_name="padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=attention_dtype,
        )
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=attention_dtype,
            check_other=False,
        )

        if padding_mask is not None:
            padding_mask = padding_mask.view(batch_size, 1, 1, key_length)

            if attention_bias is None:
                attention_bias = padding_mask
            else:
                attention_bias = attention_bias + padding_mask

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.view(1, 1, query_length, key_length)
            elif attn_mask.dim() == 3:
                if attn_mask.size(0) == self.num_heads:
                    attn_mask = attn_mask.unsqueeze(0)
                elif attn_mask.size(0) == batch_size * self.num_heads:
                    attn_mask = attn_mask.view(
                        batch_size, self.num_heads, query_length, key_length
                    )
                else:
                    raise ValueError(
                        "The first dimension of a 3D attn_mask must be "
                        "num_heads or batch_size * num_heads."
                    )
            elif attn_mask.dim() != 4:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions.")

            if attention_bias is None:
                attention_bias = attn_mask
            else:
                attention_bias = attention_bias + attn_mask

        output, weights = scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=self.dropout if self.training else 0.0,
            need_weights=need_weights,
            scale=1.0,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, query_length, embed_dim)
        output = F.linear(output, out_proj_weight, out_proj_bias)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if not need_weights:
            return output, None

        if average_attn_weights:
            weights = weights.mean(dim=1)

        return output, weights
