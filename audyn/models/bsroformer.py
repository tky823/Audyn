from typing import Union

import torch
import torch.nn as nn

from ..modules.bsrnn import (
    BandMergeModule,
    BandSplitModule,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
    MultiSourceMultiChannelBandMergeModule,
)
from ..modules.bsroformer import BandSplitRoFormerBackbone
from ..modules.normalization import RMSNorm
from .bsrnn import BandSplitRNN

__all__ = [
    "BandSplitRoFormer",
    "MultiSourceMultiChannelBandSplitRoFormer",
    "BSRoFormer",
    "MultiSourceMultiChannelBSRoFormer",
]

default_bins = [
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    4,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    12,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    24,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    48,
    128,
    129,
]


class BandSplitRoFormer(BandSplitRNN):
    """Band-split RoFormer."""

    def __init__(
        self,
        bandsplit: Union[nn.Module, BandSplitModule],
        bandmerge: Union[nn.Module, BandMergeModule],
        backbone: Union[BandSplitRoFormerBackbone, nn.Module],
    ) -> None:
        super().__init__(bandsplit, bandmerge, backbone)

    @classmethod
    def build_from_config(
        cls,
        in_channels: int,
        norm: str = "rms",
        version: Union[int, str] = "v7",
    ) -> "BandSplitRoFormer":
        from ..modules.bsroformer import BandSplitRoFormerBlock
        from .bsrnn import music_scale_bins, v7_bins

        version = str(version)

        if version.lower() in ["default"]:
            bins = default_bins
        elif version.lower() in ["7", "v7"]:
            bins = v7_bins
        elif version.lower() == "music-scale":
            bins = music_scale_bins
        else:
            raise ValueError(f"Unknown version {version} is found.")

        # band split and band merge
        embed_dim = 384
        bandmerge_hidden_channels = 1534

        # backbone
        num_heads = 8
        backbone_hidden_channels = 2048
        num_blocks = 12
        is_causal = False
        _norm = False
        dropout = 0.1
        activation = "gelu"
        eps = 1e-5
        rope_base = 10000
        share_heads = True
        norm_first = False
        bias = True

        bandsplit = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
        bandmerge = MultiChannelBandMergeModule(
            in_channels, bins, embed_dim, hidden_channels=bandmerge_hidden_channels
        )
        backbone = BandSplitRoFormerBackbone(
            embed_dim,
            num_heads,
            hidden_channels=backbone_hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=_norm,
            dropout=dropout,
            activation=activation,
            eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            norm_first=norm_first,
            bias=bias,
        )

        if norm in ["layer_norm"]:
            pass
        elif norm == "rms":
            # convert LayerNorm to RMSNorm
            for block in backbone.backbone:
                block: BandSplitRoFormerBlock
                band_block = block.band_block
                temporal_block = block.temporal_block

                norm1 = band_block.roformer.norm1
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm1.bias is None:
                    bias = False
                else:
                    bias = True

                band_block.roformer.norm1 = RMSNorm(
                    norm1.weight.size(),
                    eps=norm1.eps,
                    elementwise_affine=norm1.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm2 = band_block.roformer.norm2
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm2.bias is None:
                    bias = False
                else:
                    bias = True

                band_block.roformer.norm2 = RMSNorm(
                    norm2.weight.size(),
                    eps=norm2.eps,
                    elementwise_affine=norm2.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm1 = temporal_block.roformer.norm1
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm1.bias is None:
                    bias = False
                else:
                    bias = True

                temporal_block.roformer.norm1 = RMSNorm(
                    norm1.weight.size(),
                    eps=norm1.eps,
                    elementwise_affine=norm1.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm2 = temporal_block.roformer.norm2
                factory_kwargs = {
                    "device": norm2.weight.device,
                    "dtype": norm2.weight.dtype,
                }
                if norm2.bias is None:
                    bias = False
                else:
                    bias = True

                temporal_block.roformer.norm2 = RMSNorm(
                    norm2.weight.size(),
                    eps=norm2.eps,
                    elementwise_affine=norm2.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )
        else:
            raise ValueError(f"Unknown normalization {norm} is found.")

        model = cls(bandsplit, bandmerge, backbone)

        return model


class MultiSourceMultiChannelBandSplitRoFormer(BandSplitRoFormer):

    def __init__(
        self,
        bandsplit: Union[nn.Module, MultiChannelBandSplitModule],
        bandmerge: Union[nn.Module, MultiSourceMultiChannelBandMergeModule],
        backbone: Union[BandSplitRoFormerBackbone, nn.Module],
    ) -> None:
        super().__init__(bandsplit, bandmerge, backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiSourceMultiChannelBandSplitRoFormer.

        Args:
            input (torch.Tensor): Spectrogram of shape (*, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Separated spectrogram of shape
                (*, num_sources, in_channels, n_bins, n_frames).

        """
        x = self.bandsplit(input)
        x = self.backbone(x)
        mask = self.bandmerge(x)
        x = input.unsqueeze(dim=-4)
        output = mask * x

        return output

    @classmethod
    def build_from_config(
        cls,
        num_sources: int,
        in_channels: int,
        norm: str = "rms",
        version: Union[int, str] = "v7",
    ) -> "MultiSourceMultiChannelBandSplitRoFormer":
        from ..modules.bsroformer import BandSplitRoFormerBlock
        from .bsrnn import music_scale_bins, v7_bins

        version = str(version)

        if version.lower() in ["default"]:
            bins = default_bins
        elif version.lower() in ["7", "v7"]:
            bins = v7_bins
        elif version.lower() == "music-scale":
            bins = music_scale_bins
        else:
            raise ValueError(f"Unknown version {version} is found.")

        # band split and band merge
        embed_dim = 384
        bandmerge_hidden_channels = 1534

        # backbone
        num_heads = 8
        backbone_hidden_channels = 2048
        num_blocks = 12
        is_causal = False
        _norm = False
        dropout = 0.1
        activation = "gelu"
        eps = 1e-5
        rope_base = 10000
        share_heads = True
        norm_first = False
        bias = True

        bandsplit = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
        bandmerge = MultiSourceMultiChannelBandMergeModule(
            num_sources,
            in_channels,
            bins,
            embed_dim,
            hidden_channels=bandmerge_hidden_channels,
        )
        backbone = BandSplitRoFormerBackbone(
            embed_dim,
            num_heads,
            hidden_channels=backbone_hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=_norm,
            dropout=dropout,
            activation=activation,
            eps=eps,
            rope_base=rope_base,
            share_heads=share_heads,
            norm_first=norm_first,
            bias=bias,
        )

        if norm in ["layer_norm"]:
            pass
        elif norm == "rms":
            # convert LayerNorm to RMSNorm
            for block in backbone.backbone:
                block: BandSplitRoFormerBlock
                band_block = block.band_block
                temporal_block = block.temporal_block

                norm1 = band_block.roformer.norm1
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm1.bias is None:
                    bias = False
                else:
                    bias = True

                band_block.roformer.norm1 = RMSNorm(
                    norm1.weight.size(),
                    eps=norm1.eps,
                    elementwise_affine=norm1.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm2 = band_block.roformer.norm2
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm2.bias is None:
                    bias = False
                else:
                    bias = True

                band_block.roformer.norm2 = RMSNorm(
                    norm2.weight.size(),
                    eps=norm2.eps,
                    elementwise_affine=norm2.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm1 = temporal_block.roformer.norm1
                factory_kwargs = {
                    "device": norm1.weight.device,
                    "dtype": norm1.weight.dtype,
                }
                if norm1.bias is None:
                    bias = False
                else:
                    bias = True

                temporal_block.roformer.norm1 = RMSNorm(
                    norm1.weight.size(),
                    eps=norm1.eps,
                    elementwise_affine=norm1.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )

                norm2 = temporal_block.roformer.norm2
                factory_kwargs = {
                    "device": norm2.weight.device,
                    "dtype": norm2.weight.dtype,
                }
                if norm2.bias is None:
                    bias = False
                else:
                    bias = True

                temporal_block.roformer.norm2 = RMSNorm(
                    norm2.weight.size(),
                    eps=norm2.eps,
                    elementwise_affine=norm2.elementwise_affine,
                    bias=bias,
                    **factory_kwargs,
                )
        else:
            raise ValueError(f"Unknown normalization {norm} is found.")

        model = cls(bandsplit, bandmerge, backbone)

        return model


class BSRoFormer(BandSplitRoFormer):
    """Alias of BandSplitRoFormer."""


class MultiSourceMultiChannelBSRoFormer(MultiSourceMultiChannelBandSplitRoFormer):
    """Alias of MultiSourceMultiChannelBandSplitRoFormer."""
