from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t

from ..modules.soundstream import DecoderBlock, EncoderBlock
from .rvqvae import RVQVAE

__all__ = ["SoundStream", "Encoder", "Decoder"]


class SoundStream(RVQVAE):
    """Sound stream using residual vector quantizer.

    Args:
        encoder (nn.Module): Encoder which returns latent feature of
            shape (batch_size, embedding_dim, *).
        decoder (nn.Module): Decoder which takes latent feature of
            shape (batch_size, embedding_dim, *).
        codebook_size (int): Size of codebook.
        embedding_dim (int): Number of embedding dimension.
        num_layers (int): Number of layers of RVQ.

    """


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size_in,
            stride=1,
        )

        _in_channels = hidden_channels
        backbone = []

        for _stride in stride:
            _out_channels = depth_rate * _in_channels

            block = EncoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                num_layers=num_layers,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in, self.kernel_size_out = kernel_size_in, kernel_size_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size_in, kernel_size_out = self.kernel_size_in, self.kernel_size_out

        x = F.pad(input, (kernel_size_in // 2, kernel_size_in // 2))
        x = self.conv1d_in(x)
        x = self.backbone(x)
        x = F.pad(x, (kernel_size_out // 2, kernel_size_out // 2))
        output = self.conv1d_out(x)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        depth_rate: int,
        kernel_size_in: _size_1_t,
        kernel_size_out: _size_1_t,
        kernel_size: _size_1_t,
        stride: List[_size_1_t],
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels ** len(stride),
            kernel_size=kernel_size_in,
            stride=1,
        )

        _in_channels = hidden_channels ** len(stride)
        backbone = []

        for _stride in stride:
            _out_channels = _in_channels // depth_rate

            block = DecoderBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=_stride,
                dilation_rate=dilation_rate,
                num_layers=num_layers,
            )
            backbone.append(block)
            _in_channels = _out_channels

        self.backbone = nn.Sequential(*backbone)
        self.conv1d_out = nn.Conv1d(
            _out_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=1,
        )

        self.kernel_size_in, self.kernel_size_out = kernel_size_in, kernel_size_out

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size_in, kernel_size_out = self.kernel_size_in, self.kernel_size_out

        x = F.pad(input, (kernel_size_in // 2, kernel_size_in // 2))
        x = self.conv1d_in(x)
        x = self.backbone(x)
        x = F.pad(x, (kernel_size_out // 2, kernel_size_out // 2))
        output = self.conv1d_out(x)

        return output
