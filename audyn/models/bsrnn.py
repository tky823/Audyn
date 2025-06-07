from typing import Union

import torch
import torch.nn as nn

from ..modules.bsrnn import (
    BandMergeModule,
    BandSplitModule,
    BandSplitRNNBackbone,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
    MultiSourceMultiChannelBandMergeModule,
)

__all__ = [
    "BandSplitRNN",
    "MultiSourceMultiChannelBandSplitRNN",
    "BSRNN",
    "MultiSourceMultiChannelBSRNN",
]


v7_bins = [
    5,
    5,
    4,
    5,
    5,
    4,
    5,
    5,
    4,
    5,
    12,
    11,
    12,
    11,
    12,
    12,
    11,
    12,
    11,
    12,
    12,
    11,
    23,
    24,
    23,
    23,
    23,
    24,
    23,
    23,
    46,
    47,
    46,
    47,
    46,
    47,
    46,
    47,
    92,
    93,
    96,
]
music_scale_bins = [
    [0, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 3],
    [1, 4],
    [1, 4],
    [2, 4],
    [2, 4],
    [2, 5],
    [3, 5],
    [3, 6],
    [3, 6],
    [4, 7],
    [4, 7],
    [5, 8],
    [5, 9],
    [6, 10],
    [7, 11],
    [8, 12],
    [9, 13],
    [10, 14],
    [11, 15],
    [12, 17],
    [14, 19],
    [15, 21],
    [17, 23],
    [19, 26],
    [21, 29],
    [24, 32],
    [27, 35],
    [30, 39],
    [33, 44],
    [37, 48],
    [42, 54],
    [47, 60],
    [52, 67],
    [58, 74],
    [65, 83],
    [73, 92],
    [81, 103],
    [91, 115],
    [101, 128],
    [113, 143],
    [126, 159],
    [141, 177],
    [158, 198],
    [176, 221],
    [196, 246],
    [219, 275],
    [245, 306],
    [273, 342],
    [305, 381],
    [341, 425],
    [381, 475],
    [425, 530],
    [474, 591],
    [530, 660],
    [591, 736],
    [660, 822],
    [737, 917],
    [823, 1024],
    [918, 1025],
]


class BandSplitRNN(nn.Module):
    """Band-split RNN."""

    def __init__(
        self,
        bandsplit: Union[nn.Module, BandSplitModule, MultiChannelBandSplitModule],
        bandmerge: Union[nn.Module, BandMergeModule, MultiChannelBandMergeModule],
        backbone: Union[BandSplitRNNBackbone, nn.Module],
    ) -> None:
        super().__init__()

        self.bandsplit = bandsplit
        self.backbone = backbone
        self.bandmerge = bandmerge

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.bandsplit(input)
        x = self.backbone(x)
        mask = self.bandmerge(x)
        output = mask * input

        return output

    @classmethod
    def build_from_config(
        cls,
        in_channels: int,
        version: Union[int, str] = "v7",
    ) -> "BandSplitRNN":
        version = str(version)

        if version.lower() in ["7", "v7"]:
            bins = v7_bins
        elif version.lower() == "music-scale":
            bins = music_scale_bins
        else:
            raise ValueError(f"Unknown version {version} is found.")

        # band split and band merge
        embed_dim = 128
        bandmerge_hidden_channels = 512

        # backbone
        backbone_hidden_channels = 256
        num_blocks = 6
        is_causal = False
        norm = True
        rnn = "lstm"
        eps = 1e-5

        bandsplit = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
        bandmerge = MultiChannelBandMergeModule(
            in_channels, bins, embed_dim, hidden_channels=bandmerge_hidden_channels
        )
        backbone = BandSplitRNNBackbone(
            embed_dim,
            backbone_hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=norm,
            rnn=rnn,
            eps=eps,
        )

        model = cls(bandsplit, bandmerge, backbone)

        return model


class MultiSourceMultiChannelBandSplitRNN(BandSplitRNN):

    def __init__(
        self,
        bandsplit: Union[nn.Module, MultiChannelBandSplitModule],
        bandmerge: Union[nn.Module, MultiSourceMultiChannelBandMergeModule],
        backbone: Union[BandSplitRNNBackbone, nn.Module],
    ) -> None:
        super().__init__(bandsplit, bandmerge, backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiSourceMultiChannelBandSplitRNN.

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
        version: Union[int, str] = "v7",
    ) -> "BandSplitRNN":
        version = str(version)

        if version.lower() in ["7", "v7"]:
            bins = v7_bins
        elif version.lower() == "music-scale":
            bins = music_scale_bins
        else:
            raise ValueError(f"Unknown version {version} is found.")

        # band split and band merge
        embed_dim = 128
        bandmerge_hidden_channels = 512

        # backbone
        backbone_hidden_channels = 256
        num_blocks = 6
        is_causal = False
        norm = True
        rnn = "lstm"
        eps = 1e-5

        bandsplit = MultiChannelBandSplitModule(in_channels, bins, embed_dim)
        bandmerge = MultiSourceMultiChannelBandMergeModule(
            num_sources,
            in_channels,
            bins,
            embed_dim,
            hidden_channels=bandmerge_hidden_channels,
        )
        backbone = BandSplitRNNBackbone(
            embed_dim,
            backbone_hidden_channels,
            num_blocks=num_blocks,
            is_causal=is_causal,
            norm=norm,
            rnn=rnn,
            eps=eps,
        )

        model = cls(bandsplit, bandmerge, backbone)

        return model


class BSRNN(BandSplitRNN):
    """Alias of BandSplitRNN."""


class MultiSourceMultiChannelBSRNN(MultiSourceMultiChannelBandSplitRNN):
    """Alias of MultiSourceMultiChannelBandSplitRNN."""
