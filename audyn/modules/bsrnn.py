from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dprnn import InterChunkRNN, IntraChunkRNN
from .glu import GLU

__all__ = [
    "BandSplitModule",
    "BandMergeModule",
    "BandSplitBlock",
    "BandMergeBlock",
    "BandSplitRNNBackbone",
    "BandSplitRNNBlock",
    "IntraRNN",
    "InterRNN",
]


class _BandSplitModule(nn.Module, ABC):
    """Base class of band split module."""

    bins: Union[List[int], List[Tuple[int, int]]]
    backbone: nn.ModuleList

    def __init__(self, *args, **kwargs) -> None:
        """Define backbone."""
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandSplitModule.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Split feature of shape (*, embed_dim, len(bins), n_frames).

        """
        bins = self.bins

        n_bins = input.size(-2)

        x_stacked = []

        if isinstance(bins[0], int):
            assert sum(bins) == n_bins

            x = torch.split(input, bins, dim=-2)

            for band_idx in range(len(bins)):
                x_band = x[band_idx]
                block = self.backbone[band_idx]
                x_band = block(x_band)
                x_stacked.append(x_band)
        else:
            for band_idx in range(len(bins)):
                start_bin, end_bin = bins[band_idx]
                _, x_band, _ = torch.split(
                    input, [start_bin, end_bin - start_bin, n_bins - end_bin], dim=-2
                )
                block = self.backbone[band_idx]
                x_band = block(x_band)
                x_stacked.append(x_band)

        output = torch.stack(x_stacked, dim=-2)

        return output


class _BandMergeModule(nn.Module):
    """Base class of band merge module."""

    bins: Union[List[int], List[Tuple[int, int]]]
    backbone: nn.ModuleList
    frequency_assignment: Optional[torch.Tensor]

    def __init__(self, *args, **kwargs) -> None:
        """Define backbone."""
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandMergeModule.

        Args:
            input (torch.Tensor): Split feature of shape  (*, embed_dim, len(bins), n_frames).

        Returns:
            torch.Tensor: Spectrogram-like feature of shape (*, n_bins, n_frames).

        """
        bins = self.bins
        frequency_assignment = self.frequency_assignment

        n_bands = input.size(-2)
        x = torch.unbind(input, dim=-2)
        x_stacked = []

        if isinstance(bins[0], int):
            assert len(bins) == n_bands

            for band_idx in range(n_bands):
                x_band = x[band_idx]
                block = self.backbone[band_idx]
                x_band = block(x_band)
                x_stacked.append(x_band)

            output = torch.cat(x_stacked, dim=-2)
        else:
            n_bins = frequency_assignment.size(-1)

            for band_idx in range(n_bands):
                start_idx, end_idx = bins[band_idx]
                x_band = x[band_idx]
                block = self.backbone[band_idx]
                x_band = block(x_band)
                x_band = F.pad(x_band, (0, 0, start_idx, n_bins - end_idx))
                x_stacked.append(x_band)

            x_stacked = torch.stack(x_stacked, dim=-3)
            x_stacked = frequency_assignment.unsqueeze(dim=-1) * x_stacked
            output = x_stacked.sum(dim=-3)

        return output

    @staticmethod
    def build_frequency_assignment(
        bins: List[Tuple[int, int]],
        n_bins: Optional[int] = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> torch.Tensor:
        """Build frequency assignment to bands.

        Args:
            bins (list): List of (start, end) in each band.
            n_bins (int, optional): Number of frequency bins.

        Returns:
            torch.Tensor: Frequency assignment of shape (n_bands, n_bins).

        """
        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

        if n_bins is None:
            last_band = bins[-1]
            _, n_bins = last_band

        n_bands = len(bins)
        assignment = torch.zeros((n_bands, n_bins), **factory_kwargs)

        for band_idx, (start_bin, end_bin) in enumerate(bins):
            assignment[band_idx, start_bin:end_bin] = 1

        assignment = assignment / assignment.sum(dim=0)

        return assignment


class BandSplitModule(_BandSplitModule):
    """Band split module."""

    def __init__(self, bins: Union[List[int], List[Tuple[int, int]]], embed_dim: int) -> None:
        super().__init__()

        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for _bins in bins:
            if isinstance(_bins, int):
                n_bins = _bins
            else:
                start_bin, end_bin = _bins
                n_bins = end_bin - start_bin

            block = BandSplitBlock(n_bins, embed_dim)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


class BandMergeModule(_BandMergeModule):
    """Band merge module."""

    def __init__(
        self,
        bins: Union[List[int], List[Tuple[int, int]]],
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        if isinstance(bins[0], int):
            frequency_overlap = False
        else:
            frequency_overlap = True

        max_bin = 0

        for _bins in bins:
            if frequency_overlap:
                start_bin, end_bin = _bins
                n_bins = end_bin - start_bin
                max_bin = end_bin
            else:
                n_bins = _bins
                max_bin += _bins

            block = BandMergeBlock(
                n_bins,
                embed_dim,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        if frequency_overlap:
            frequency_assignment = self.build_frequency_assignment(bins, n_bins=max_bin)
        else:
            frequency_assignment = None

        self.register_buffer("frequency_assignment", frequency_assignment, persistent=False)


class MultiChannelBandSplitModule(_BandSplitModule):
    """Band split module for multichannel input."""

    def __init__(
        self,
        in_channels: int,
        bins: Union[List[int], List[Tuple[int, int]]],
        embed_dim: int,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for _bins in bins:
            if isinstance(_bins, int):
                n_bins = _bins
            else:
                start_bin, end_bin = _bins
                n_bins = end_bin - start_bin

            block = MultiChannelBandSplitBlock(in_channels, n_bins, embed_dim)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


class MultiChannelBandMergeModule(_BandMergeModule):
    """Band merge module for multichannel output."""

    def __init__(
        self,
        out_channels: int,
        bins: Union[List[int], List[Tuple[int, int]]],
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        if isinstance(bins[0], int):
            frequency_overlap = False
        else:
            frequency_overlap = True

        max_bin = 0

        for _bins in bins:
            if frequency_overlap:
                start_bin, end_bin = _bins
                n_bins = end_bin - start_bin
                max_bin = end_bin
            else:
                n_bins = _bins
                max_bin += _bins

            block = MultiChannelBandMergeBlock(
                out_channels,
                n_bins,
                embed_dim,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        if frequency_overlap:
            frequency_assignment = self.build_frequency_assignment(bins, n_bins=max_bin)
        else:
            frequency_assignment = None

        self.register_buffer("frequency_assignment", frequency_assignment, persistent=False)


class BandSplitBlock(nn.Module):
    """BandSplitBlock composed of layer norm and linear.

    Args:
        n_bins (int): Number of bins in band.
        embed_dim (int): Embedding dimension.

    """

    def __init__(self, n_bins: int, embed_dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(2 * n_bins)
        self.linear = nn.Linear(2 * n_bins, embed_dim)

        self.n_bins = n_bins
        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandSplitBlock.

        Args:
            input (torch.Tensor): Complex spectrogram of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Transformed feature of shape (*, embed_dim, n_frames).

        """
        assert torch.is_complex(input), "Complex spectrogram is expected."

        embed_dim = self.embed_dim
        *batch_shape, n_bins, n_frames = input.size()

        x = input.contiguous()
        x = x.view(-1, n_bins, n_frames)
        x = torch.view_as_real(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, n_frames, n_bins * 2)
        x = self.norm(x)
        x = self.linear(x)
        x = x.permute(0, 2, 1).contiguous()
        output = x.view(*batch_shape, embed_dim, n_frames)

        return output


class BandMergeBlock(nn.Module):
    """BandMergeBlock composed of layer norm and multi-layer perceptron (MLP).

    Args:
        n_bins (int): Number of bins in band.
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of layers in MLP.

    The implementation is based on [#luo2023music]_.

    .. [#luo2023music]
        Y. Luo et al., "Music source separation with band-split RNN,"
        *IEEE/ACM Transactions on Audio, Speech, and Language Processing*,
        vol. 31, pp.1893-1901, 2023.

    """

    def __init__(
        self,
        n_bins: int,
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        mlp = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                in_channels = embed_dim
            else:
                in_channels = hidden_channels

            if layer_idx == num_layers - 1:
                out_channels = 2 * n_bins
            else:
                out_channels = hidden_channels

            mlp.append(nn.Linear(in_channels, out_channels))

            if layer_idx == num_layers - 1:
                mlp.append(GLU(out_channels, out_channels))
            else:
                mlp.append(nn.Tanh())

        self.mlp = nn.Sequential(*mlp)

        self.n_bins = n_bins
        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandMergeBlock.

        Args:
            input (torch.Tensor): Band feature of shape (*, embed_dim, n_frames).

        Returns:
            torch.Tensor: Merged complex feature of shape (*, n_bins, n_frames).

        """
        n_bins = self.n_bins
        *batch_shape, embed_dim, n_frames = input.size()

        x = input.contiguous()
        x = x.view(-1, embed_dim, n_frames)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = self.mlp(x)
        x = x.view(-1, n_frames, n_bins, 2)
        x = torch.view_as_complex(x)
        x = x.permute(0, 2, 1).contiguous()
        output = x.view(*batch_shape, n_bins, n_frames)

        return output


class MultiChannelBandSplitBlock(nn.Module):
    """MultiChannelBandSplitBlock composed of layer norm and linear.

    Args:
        in_channels (int): Number of input channels.
        n_bins (int): Number of bins in band.
        embed_dim (int): Embedding dimension.

    """

    def __init__(self, in_channels: int, n_bins: int, embed_dim: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(2 * in_channels * n_bins)
        self.linear = nn.Linear(2 * in_channels * n_bins, embed_dim)

        self.in_channels = in_channels
        self.n_bins = n_bins
        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiChannelBandSplitBlock.

        Args:
            input (torch.Tensor): Complex spectrogram of shape (*, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Transformed feature of shape (*, embed_dim, n_frames).

        """
        assert torch.is_complex(input), "Complex spectrogram is expected."

        embed_dim = self.embed_dim
        *batch_shape, in_channels, n_bins, n_frames = input.size()

        x = input.contiguous()
        x = x.view(-1, in_channels * n_bins, n_frames)
        x = torch.view_as_real(x)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(-1, n_frames, in_channels * n_bins * 2)
        x = self.norm(x)
        x = self.linear(x)
        x = x.permute(0, 2, 1).contiguous()
        output = x.view(*batch_shape, embed_dim, n_frames)

        return output


class MultiChannelBandMergeBlock(nn.Module):
    """MultiChannelBandMergeBlock composed of layer norm and multi-layer perceptron (MLP).

    Args:
        out_channels (int): Number of output channels.
        n_bins (int): Number of bins in band.
        embed_dim (int): Embedding dimension.
        num_layers (int): Number of layers in MLP.

    The implementation is based on [#luo2023music]_.

    .. [#luo2023music]
        Y. Luo et al., "Music source separation with band-split RNN,"
        *IEEE/ACM Transactions on Audio, Speech, and Language Processing*,
        vol. 31, pp.1893-1901, 2023.

    """

    def __init__(
        self,
        out_channels: int,
        n_bins: int,
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(embed_dim)

        mlp = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                _in_channels = embed_dim
            else:
                _in_channels = hidden_channels

            if layer_idx == num_layers - 1:
                _out_channels = 2 * out_channels * n_bins
            else:
                _out_channels = hidden_channels

            mlp.append(nn.Linear(_in_channels, _out_channels))

            if layer_idx == num_layers - 1:
                mlp.append(GLU(_out_channels, _out_channels))
            else:
                mlp.append(nn.Tanh())

        self.mlp = nn.Sequential(*mlp)

        self.out_channels = out_channels
        self.n_bins = n_bins
        self.embed_dim = embed_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiChannelBandMergeBlock.

        Args:
            input (torch.Tensor): Band feature of shape (*, embed_dim, n_frames).

        Returns:
            torch.Tensor: Merged complex feature of shape (*, out_channels, n_bins, n_frames).

        """
        out_channels = self.out_channels
        n_bins = self.n_bins
        *batch_shape, embed_dim, n_frames = input.size()

        x = input.contiguous()
        x = x.view(-1, embed_dim, n_frames)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm(x)
        x = self.mlp(x)
        x = x.view(-1, n_frames, out_channels * n_bins, 2)
        x = torch.view_as_complex(x)
        x = x.permute(0, 2, 1).contiguous()
        output = x.view(*batch_shape, out_channels, n_bins, n_frames)

        return output


class BandSplitRNNBackbone(nn.Module):
    """Backbone of BandSplitRNN."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        num_blocks: int = 6,
        is_causal: bool = False,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "lstm",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        backbone = []

        for _ in range(num_blocks):
            backbone.append(
                BandSplitRNNBlock(
                    num_features,
                    hidden_channels,
                    norm=norm,
                    is_causal=is_causal,
                    rnn=rnn,
                    eps=eps,
                )
            )

        self.backbone = nn.Sequential(*backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BandSplitRNNBackbone.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        output = self.backbone(input)

        return output


class BandSplitRNNBlock(nn.Module):
    """RNN block for band and sequence modeling."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.band_block = IntraRNN(num_features, hidden_channels, norm=norm, rnn=rnn, eps=eps)
        self.temporal_block = InterRNN(
            num_features, hidden_channels, norm=norm, is_causal=is_causal, rnn=rnn, eps=eps
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RNN block for BandSplitRNN.

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


class IntraRNN(IntraChunkRNN):
    """RNN for band modeling in BandSplitRNN."""


class InterRNN(InterChunkRNN):
    """RNN for sequence modeling in BandSplitRNN."""
