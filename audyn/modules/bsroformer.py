from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.modules.activation import NonDynamicallyQuantizableLinear

from .activation import (
    RotaryPositionalMultiheadAttention as _RotaryPositionalMultiheadAttention,
)
from .positional_encoding import RotaryPositionalEmbedding
from .tasnet import get_layer_norm
from .transformer import get_activation

__all__ = [
    "BandSplitRoFormerBackbone",
    "BandSplitRoFormerBlock",
    "IntraChunkRoFormer",
    "InterChunkRoFormer",
    "IntraRoFormer",
    "InterRoFormer",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class BandSplitRoFormerBackbone(nn.Module):
    """Backbone of BandSplitRoFormer."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        head_channels: int,
        hidden_channels: int,
        num_blocks: int = 12,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        backbone = []

        for _ in range(num_blocks):
            backbone.append(
                BandSplitRoFormerBlock(
                    num_features,
                    num_heads,
                    head_channels,
                    hidden_channels,
                    is_causal=is_causal,
                    norm=norm,
                    dropout=dropout,
                    activation=activation,
                    eps=eps,
                    rope_base=rope_base,
                    share_heads=share_heads,
                    norm_first=norm_first,
                    bias=bias,
                )
            )

        self.backbone = nn.Sequential(*backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandSplitRoFormerBackbone.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        output = self.backbone(input)

        return output


class BandSplitRoFormerBlock(nn.Module):
    """RoFormer block for band and sequence modeling."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        head_channels: int,
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.band_block = IntraRoFormer(
            num_features,
            num_heads,
            head_channels,
            hidden_channels,
            norm=norm,
            dropout=dropout,
            activation=activation,
            eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            norm_first=norm_first,
            bias=bias,
        )
        self.temporal_block = InterRoFormer(
            num_features,
            num_heads,
            head_channels,
            hidden_channels,
            is_causal=is_causal,
            norm=norm,
            dropout=dropout,
            activation=activation,
            eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            norm_first=norm_first,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RoFormer block for BandSplitRoFormer.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, n_bands, n_frames).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, n_bands, n_frames).

        """
        x = self.band_block(input)
        output = self.temporal_block(x)

        return output


class IntraChunkRoFormer(nn.Module):
    """Intra-chunk dual-path RoFormer."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        head_channels: int,
        hidden_channels: int,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.head_channels = head_channels
        self.hidden_channels = hidden_channels

        self.roformer = RoFormerEncoderLayer(
            num_heads * head_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=True,
            norm_first=norm_first,
            qdim=num_features,
            kdim=num_features,
            vdim=num_features,
            bias=bias,
        )

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = get_layer_norm(norm, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm = "gLN"
                self.norm = get_layer_norm(norm, num_features, is_causal=False, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of intra-chunk dual-path RoFormer block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        num_features = self.num_features
        batch_size, _, inter_length, chunk_size = input.size()

        residual = input
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size * inter_length, chunk_size, num_features)
        x = self.roformer(x)
        x = x.view(batch_size, inter_length * chunk_size, num_features)
        x = x.permute(0, 2, 1).contiguous()

        if self.norm is not None:
            x = self.norm(x)

        x = x.view(batch_size, num_features, inter_length, chunk_size)
        output = x + residual

        return output


class InterChunkRoFormer(nn.Module):
    """Inter-chunk dual-path RoFormer."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        head_channels: int,
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.head_channels = head_channels
        self.hidden_channels = hidden_channels
        self.is_causal = is_causal

        self.roformer = RoFormerEncoderLayer(
            num_heads * head_channels,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=True,
            norm_first=norm_first,
            qdim=num_features,
            kdim=num_features,
            vdim=num_features,
            bias=bias,
        )

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = get_layer_norm(norm, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm = "gLN"
                self.norm = get_layer_norm(norm, num_features, is_causal=False, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of inter-chunk dual-path RoFormer block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        num_features = self.num_features
        is_causal = self.is_causal
        batch_size, _, inter_length, chunk_size = input.size()

        residual = input
        x = input.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * chunk_size, inter_length, num_features)
        x = self.roformer(x, is_causal=is_causal)
        x = x.view(batch_size, chunk_size * inter_length, num_features)
        x = x.permute(0, 2, 1).contiguous()

        if self.norm is not None:
            x = self.norm(x)

        x = x.view(batch_size, num_features, chunk_size, inter_length)
        x = x.permute(0, 1, 3, 2).contiguous()
        output = x + residual

        return output


class RoFormerEncoderLayer(nn.Module):
    """Encoder layer of RoFormer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = True,
        norm_first: bool = True,
        qdim: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        bias: Optional[bool] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        if qdim is None:
            qdim = d_model

        self.self_attn = RotaryPositionalMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            share_heads=share_heads,
            batch_first=batch_first,
            qdim=qdim,
            kdim=kdim,
            vdim=vdim,
            base=rope_base,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(qdim, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, qdim, bias=bias, **factory_kwargs)

        self.norm_first = norm_first

        if norm == "layer_norm":
            if bias is None:
                bias = True

            if IS_TORCH_LT_2_1:
                assert bias, "Only bias=True is supported for torch < 2.1."

                layer_norm_kwargs = {}
            else:
                layer_norm_kwargs = {"bias": bias}

            self.norm1 = nn.LayerNorm(
                qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
            )
            self.norm2 = nn.LayerNorm(
                qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
            )
        elif norm == "rms":
            if hasattr(nn, "RMSNorm"):
                if bias is None:
                    layer_norm_kwargs = {}
                elif bias:
                    raise ValueError("bias=True is not supported for nn.RMSNorm.")
                else:
                    layer_norm_kwargs = {}

                self.norm1 = nn.RMSNorm(
                    qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
                )
                self.norm2 = nn.RMSNorm(
                    qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
                )
            else:
                from .normalization import RMSNorm

                if bias is None:
                    layer_norm_kwargs = {}
                else:
                    layer_norm_kwargs = {"bias": bias}

                self.norm1 = RMSNorm(
                    qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
                )
                self.norm2 = RMSNorm(
                    qdim, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
                )
        else:
            raise ValueError(f"Unsupported norm type {norm} is found.")

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (torch.Tensor): the sequence to the encoder layer.
            src_mask (torch.BoolTensor, optional): the mask for the src sequence.
            src_key_padding_mask (torch.BoolTensor, optional): the mask for the src keys
                per batch.
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)


class RotaryPositionalMultiheadAttention(_RotaryPositionalMultiheadAttention):
    """Multi-head attention with rotary positional encoding for BSRoFormer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: Optional[bool] = None,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        qdim: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        base: int = 10000,
        share_heads: bool = True,
        batch_first: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super(nn.MultiheadAttention, self).__init__()

        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        self.embed_dim = embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads"
        )

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.qdim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, self.qdim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias is None:
            bias = False

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, qdim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")
        else:
            self.bias_k = self.bias_v = None

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self._reset_parameters()

        self.rope = RotaryPositionalEmbedding(base=base, batch_first=batch_first)

        self.share_heads = share_heads


class IntraRoFormer(IntraChunkRoFormer):
    """RoFormer for band modeling in Band-split RoFormer."""


class InterRoFormer(InterChunkRoFormer):
    """RoFormer for sequence modeling in Band-split RoFormer."""
