from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.roformer import RoFormerEncoderLayer
from .tasnet import get_layer_norm

__all__ = [
    "BandSplitRoFormerBackbone",
    "BandSplitRoFormerBlock",
    "IntraChunkRoFormer",
    "InterChunkRoFormer",
    "IntraRoFormer",
    "InterRoFormer",
]


class BandSplitRoFormerBackbone(nn.Module):
    """Backbone of BandSplitRoFormer."""

    def __init__(
        self,
        num_features: int,
        num_heads: int,
        hidden_channels: int,
        num_blocks: int = 12,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        backbone = []

        for _ in range(num_blocks):
            backbone.append(
                BandSplitRoFormerBlock(
                    num_features,
                    num_heads,
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
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.band_block = IntraRoFormer(
            num_features,
            num_heads,
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
        hidden_channels: int,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_channels = hidden_channels

        self.roformer = RoFormerEncoderLayer(
            num_features,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=True,
            norm_first=norm_first,
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
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[
            Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = False,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        norm_first: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_channels = hidden_channels
        self.is_causal = is_causal

        self.roformer = RoFormerEncoderLayer(
            num_features,
            nhead=num_heads,
            dim_feedforward=hidden_channels,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            batch_first=True,
            norm_first=norm_first,
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


class IntraRoFormer(IntraChunkRoFormer):
    """RoFormer for band modeling in Band-split RoFormer."""


class InterRoFormer(InterChunkRoFormer):
    """RoFormer for sequence modeling in Band-split RoFormer."""
