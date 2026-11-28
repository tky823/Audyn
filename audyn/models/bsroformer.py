import os
from typing import Union

import hydra
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..modules.bsrnn import (
    AudioSeparatorCompatibleBandMergeModule,
    AudioSeparatorCompatibleBandSplitModule,
    BandMergeModule,
    BandSplitModule,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
    MultiSourceMultiChannelBandMergeModule,
)
from ..modules.bsroformer import (
    AudioSeparatorCompatibleBandSplitRoFormerBackbone,
    BandSplitRoFormerBackbone,
)
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
    """Band-split RoFormer.

    Args:
        bandsplit (nn.Module or BandSplitModule): Module for band-splitting.
        bandmerge (nn.Module or BandMergeModule): Module for band-merging.
        backbone (nn.Module or BandSplitRoFormerBackbone): Backbone module for RoFormer.

    """

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
        version: Union[int, str] = "default",
    ) -> "BandSplitRoFormer":
        """Build Band-split RoFormer from configuration.

        Args:
            in_channels (int): Number of input channels.
            version (Union[int, str]): Version of the model. Default is ``"default"``.

        Returns:
            BandSplitRoFormer: Built Band-split RoFormer.

        """
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
        bandmerge_hidden_channels = 1536

        # backbone
        num_heads = 8
        backbone_head_channels = 64
        backbone_hidden_channels = 1536
        num_blocks = 12
        is_causal = False
        _norm = False
        dropout = 0.1
        activation = "gelu"
        eps = 1e-5
        rope_base = 10000
        share_heads = True
        norm_first = True
        bias = False

        bandsplit = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
        bandmerge = MultiChannelBandMergeModule(
            in_channels, bins, embed_dim, hidden_channels=bandmerge_hidden_channels
        )
        backbone = BandSplitRoFormerBackbone(
            embed_dim,
            num_heads,
            head_channels=backbone_head_channels,
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
        model = cls(bandsplit, bandmerge, backbone)

        return model


class AudioSeparatorCompatibleBandSplitRoFormer(BandSplitRoFormer):
    """Band-split RoFormer compatible with audio-separator.

    Args:
        bandsplit (nn.Module or BandSplitModule): Module for band-splitting.
        bandmerge (nn.Module or BandMergeModule): Module for band-merging.
        backbone (nn.Module or AudioSeparatorCompatibleBandSplitRoFormerBackbone): Backbone module
            for RoFormer.

    """

    @classmethod
    def build_from_config(
        cls,
        in_channels: int,
        version: Union[int, str] = "mel-scale",
    ) -> "AudioSeparatorCompatibleBandSplitRoFormer":
        """Build Band-split RoFormer from configuration.

        Args:
            in_channels (int): Number of input channels.
            version (int or str): Version of the model. Default is ``"mel-scale"``.

        Returns:
            AudioSeparatorCompatibleBandSplitRoFormer: Built Band-split RoFormer.

        """
        from .bsrnn import mel_scale_bins, music_scale_bins, v7_bins

        version = str(version)

        if version.lower() in ["default"]:
            bins = default_bins
        elif version.lower() in ["7", "v7"]:
            bins = v7_bins
        elif version.lower() == "music-scale":
            bins = music_scale_bins
        elif version.lower() == "mel-scale":
            bins = mel_scale_bins
        else:
            raise ValueError(f"Unknown version {version} is found.")

        # band split and band merge
        embed_dim = 384
        bandmerge_hidden_channels = 1536
        bandsplit_norm = "rms"
        bandsplit_eps = 1e-12
        bandmerge_eps = 1e-12

        # backbone
        num_heads = 8
        backbone_head_channels = 64
        backbone_hidden_channels = 1536
        num_blocks = 6
        is_causal = False
        backbone_norm = "rms"
        dropout = 0.0
        activation = "gelu"
        backbone_eps = 1e-12
        rope_base = 10000
        share_heads = True
        norm_first = True
        bias = False

        bandsplit = AudioSeparatorCompatibleBandSplitModule(
            in_channels, bins, embed_dim, norm=bandsplit_norm, eps=bandsplit_eps
        )
        bandmerge = AudioSeparatorCompatibleBandMergeModule(
            in_channels,
            bins,
            embed_dim,
            hidden_channels=bandmerge_hidden_channels,
            norm=False,
            eps=bandmerge_eps,
        )
        backbone = AudioSeparatorCompatibleBandSplitRoFormerBackbone(
            embed_dim,
            num_heads,
            head_channels=backbone_head_channels,
            hidden_channels=backbone_hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=backbone_norm,
            dropout=dropout,
            activation=activation,
            eps=backbone_eps,
            rope_base=rope_base,
            share_heads=share_heads,
            norm_first=norm_first,
            bias=bias,
        )
        model = cls(bandsplit, bandmerge, backbone)

        return model

    @classmethod
    def build_from_pretrained(
        cls, pretrained_model_name_or_path: str
    ) -> "AudioSeparatorCompatibleBandSplitRoFormer":
        # to avoid circulat import
        from ..utils import model_cache_dir
        from ..utils._github import download_file_from_github_release

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path, map_location=lambda storage, loc: storage
            )

            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            model: AudioSeparatorCompatibleBandSplitRoFormer = hydra.utils.instantiate(
                resolved_config.model
            )
            model.load_state_dict(state_dict["model"])
        elif pretrained_model_name_or_path == "audio-separator_melbsroformer":
            url = "https://github.com/tky823/Audyn/releases/download/v0.4.0/audio-separator_melbsroformer.pth"  # noqa: E501
            filename = os.path.basename(url)
            path = os.path.join(
                model_cache_dir,
                "AudioSeparatorCompatibleBandSplitRoFormer",
                "e4704501",
                filename,
            )
            download_file_from_github_release(url, path)

            model = cls.build_from_pretrained(path)
        else:
            raise ValueError(f"{pretrained_model_name_or_path} is not found.")

        return model


class MultiSourceMultiChannelBandSplitRoFormer(BandSplitRoFormer):
    """Band-split RoFormer for multi-source and multi-channel input.

    Args:
        bandsplit (nn.Module or MultiChannelBandSplitModule): Module for band-splitting.
        bandmerge (nn.Module or MultiSourceMultiChannelBandMergeModule): Module for band-merging.
        backbone (nn.Module or BandSplitRoFormerBackbone): Backbone module for RoFormer.

    """

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
            input (torch.Tensor): Spectrogram of shape ``(*, in_channels, n_bins, n_frames)``.

        Returns:
            torch.Tensor: Separated spectrogram of shape
                ``(*, num_sources, in_channels, n_bins, n_frames)``.

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
        version: Union[int, str] = "default",
    ) -> "MultiSourceMultiChannelBandSplitRoFormer":
        """Build Band-split RoFormer from configuration.

        Args:
            in_channels (int): Number of input channels.
            version (Union[int, str]): Version of the model. Default is ``"default"``.

        Returns:
            MultiSourceMultiChannelBandSplitRoFormer: Built Band-split RoFormer.

        """
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
        bandmerge_hidden_channels = 1536

        # backbone
        num_heads = 8
        backbone_head_channels = 64
        backbone_hidden_channels = 1536
        num_blocks = 12
        is_causal = False
        _norm = False
        dropout = 0.1
        activation = "gelu"
        eps = 1e-5
        rope_base = 10000
        share_heads = True
        norm_first = True
        bias = False

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
            head_channels=backbone_head_channels,
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

        model = cls(bandsplit, bandmerge, backbone)

        return model


class BSRoFormer(BandSplitRoFormer):
    """Alias of BandSplitRoFormer."""


class MultiSourceMultiChannelBSRoFormer(MultiSourceMultiChannelBandSplitRoFormer):
    """Alias of MultiSourceMultiChannelBandSplitRoFormer."""
