from typing import List

import torch
import torch.nn as nn

__all__ = [
    "BandSplitModule",
    "BandTransformBlock",
]


class BandSplitModule(nn.Module):
    def __init__(self, bins: List[int], embed_dim: int) -> None:
        super().__init__()

        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for n_bins in bins:
            block = BandTransformBlock(n_bins, embed_dim)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandSplitModule.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Split feature of shape (*, embed_dim, len(bins), n_frames).

        """
        bins = self.bins
        embed_dim = self.embed_dim

        *batch_shape, n_bins, n_frames = input.size()

        x = input.view(-1, n_bins, n_frames)
        x = torch.split(x, bins, dim=-2)

        x_stacked = []

        for band_idx in range(len(bins)):
            x_band = x[band_idx]
            block = self.backbone[band_idx]
            x_band = block(x_band)
            x_stacked.append(x_band)

        x = torch.stack(x_stacked, dim=-2)
        output = x.view(*batch_shape, embed_dim, len(bins), n_frames)

        return output


class BandTransformBlock(nn.Module):
    """BandTransformBlock composed of layer norm and linear.

    Args:
        n_bins (int): Number of bins in band.
        embed_dim (int): Embedding dimension.

    """

    def __init__(self, n_bins: int, embed_dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(2 * n_bins)
        self.linear = nn.Linear(2 * n_bins, embed_dim)

        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandTransformBlock.

        Args:
            input (torch.Tensor): Band of complex spectrogram of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Transformed feature of shape (*, embed_dim, n_frames).

        """
        assert torch.is_complex(input), "Complex spectrogram is expected."

        embed_dim = self.embed_dim
        *batch_shape, n_bins, n_frames = input.size()

        x = input.view(-1, n_bins, n_frames)
        x = torch.view_as_real(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, n_bins * 2)
        x = self.norm(x)
        x = self.linear(x)
        x = x.view(-1, n_frames, embed_dim)
        x = x.permute(0, 2, 1).contiguous()
        output = x.view(*batch_shape, embed_dim, n_frames)

        return output
