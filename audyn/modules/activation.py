from typing import Optional, Tuple

import torch
import torch.nn as nn
from packaging import version

__all__ = [
    "MultiheadSelfAttention",
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
        key_padding_mask: Optional[torch.Tensor] = None,
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
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
            **kwargs,
        )
