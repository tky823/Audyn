from abc import ABC
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .dprnn import get_rnn
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

    bins: List[int]
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

        assert sum(bins) == n_bins

        x = torch.split(input, bins, dim=-2)
        x_stacked = []

        for band_idx in range(len(bins)):
            x_band = x[band_idx]
            block = self.backbone[band_idx]
            x_band = block(x_band)
            x_stacked.append(x_band)

        output = torch.stack(x_stacked, dim=-2)

        return output


class _BandMergeModule(nn.Module):
    """Base class of band merge module."""

    bins: List[int]
    backbone: nn.ModuleList

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

        n_bands = input.size(-2)

        assert len(bins) == n_bands

        x = torch.unbind(input, dim=-2)
        x_stacked = []

        for band_idx in range(n_bands):
            x_band = x[band_idx]
            block = self.backbone[band_idx]
            x_band = block(x_band)
            x_stacked.append(x_band)

        output = torch.cat(x_stacked, dim=-2)

        return output


class BandSplitModule(_BandSplitModule):
    """Band split module."""

    def __init__(self, bins: List[int], embed_dim: int) -> None:
        super().__init__()

        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for n_bins in bins:
            block = BandSplitBlock(n_bins, embed_dim)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


class BandMergeModule(_BandMergeModule):
    """Band merge module."""

    def __init__(
        self,
        bins: List[int],
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for n_bins in bins:
            block = BandMergeBlock(
                n_bins,
                embed_dim,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


class MultiChannelBandSplitModule(_BandSplitModule):
    """Band split module for multichannel input."""

    def __init__(self, in_channels: int, bins: List[int], embed_dim: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for n_bins in bins:
            block = MultiChannelBandSplitBlock(in_channels, n_bins, embed_dim)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


class MultiChannelBandMergeModule(_BandMergeModule):
    """Band merge module for multichannel output."""

    def __init__(
        self,
        out_channels: int,
        bins: List[int],
        embed_dim: int,
        hidden_channels: int = 512,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        self.bins = bins
        self.embed_dim = embed_dim

        backbone = []

        for n_bins in bins:
            block = MultiChannelBandMergeBlock(
                out_channels,
                n_bins,
                embed_dim,
                hidden_channels=hidden_channels,
                num_layers=num_layers,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)


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


class IntraRNN(nn.Module):
    """RNN for band modeling in BandSplitRNN."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_channels = hidden_channels

        num_directions = 2

        self.rnn = get_rnn(
            rnn,
            input_size=num_features,
            hidden_size=hidden_channels,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            assert norm.upper() == "LN", "Only layer normalization is supported."

            self.norm = nn.LayerNorm(num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                self.norm = nn.LayerNorm(num_features, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of band RNN block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, n_bands, n_frames).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, n_bands, n_frames)

        """
        num_features = self.num_features
        batch_size, _, n_bands, n_frames = input.size()

        self.rnn.flatten_parameters()

        residual = input
        x = input.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * n_frames, n_bands, num_features)

        if self.norm is not None:
            x = self.norm(x)

        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.view(batch_size, n_frames, n_bands, num_features)
        x = x.permute(0, 3, 2, 1)

        output = x + residual

        return output


class InterRNN(nn.Module):
    """RNN for sequence modeling in BandSplitRNN."""

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

        self.num_features = num_features
        self.hidden_channels = hidden_channels

        if is_causal:
            # uni-direction
            num_directions = 1
            self.rnn = get_rnn(
                rnn,
                input_size=num_features,
                hidden_size=hidden_channels,
                batch_first=True,
                bidirectional=False,
            )
        else:
            # bi-direction
            num_directions = 2
            self.rnn = get_rnn(
                rnn,
                input_size=num_features,
                hidden_size=hidden_channels,
                batch_first=True,
                bidirectional=True,
            )

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            assert norm.upper() == "LN", "Only layer normalization is supported."

            self.norm = nn.LayerNorm(num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                self.norm = nn.LayerNorm(num_features, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of temporal RNN block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, n_bands, n_frames).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, n_bands, n_frames).

        """
        num_features = self.num_features
        batch_size, _, n_bands, n_frames = input.size()

        self.rnn.flatten_parameters()

        residual = input
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size * n_bands, n_frames, num_features)

        if self.norm is not None:
            x = self.norm(x)

        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.view(batch_size, n_bands, n_frames, num_features)
        x = x.permute(0, 3, 1, 2)

        output = x + residual

        return output
