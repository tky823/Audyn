from typing import Union

import torch
import torch.nn as nn

from ..modules.bsrnn import (
    BandMergeModule,
    BandSplitModule,
    BandSplitRNNBackbone,
    MultiChannelBandMergeModule,
    MultiChannelBandSplitModule,
)

__all__ = [
    "BandSplitRNN",
    "BSRNN",
]


class BandSplitRNN(nn.Module):
    """BandSplitRNN."""

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


class BSRNN(BandSplitRNN):
    """Alias of BandSplitRNN."""
