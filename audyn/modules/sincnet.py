import torch
import torch.nn as nn
from torch.nn.common_types import _size_1_t

from .filterbanks import QuadratureSincConv1d as _QuadratureSincConv1d


class _ConvBlock1d(nn.Module):
    def __init__(
        self,
        conv1d: nn.Module,
        pool1d: nn.Module,
        norm1d: nn.Module,
        activation: nn.Module,
    ) -> None:
        super().__init__()

        self.conv1d = conv1d
        self.pool1d = pool1d
        self.norm1d = norm1d
        self.activation = activation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(input)
        x = self.pool1d(x)
        x = self.norm1d(x)
        output = self.activation(x)

        return output


class QuadratureSincConvBlock1d(_ConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        pool_size: _size_1_t = 3,
        stride: _size_1_t = 1,
        sample_rate: int = 16000,
    ) -> None:
        assert in_channels == 1

        conv1d = QuadratureSincConv1d(
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            sample_rate=sample_rate,
        )
        pool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        norm1d = nn.InstanceNorm1d(out_channels, affine=True)
        activation = nn.LeakyReLU()

        super().__init__(
            conv1d=conv1d,
            pool1d=pool1d,
            norm1d=norm1d,
            activation=activation,
        )


class ConvBlock1d(_ConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        pool_size: _size_1_t = 3,
        stride: _size_1_t = 1,
    ) -> None:
        conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        pool1d = nn.MaxPool1d(pool_size, stride=pool_size)
        norm1d = nn.InstanceNorm1d(out_channels, affine=True)
        activation = nn.LeakyReLU()

        super().__init__(
            conv1d=conv1d,
            pool1d=pool1d,
            norm1d=norm1d,
            activation=activation,
        )


class QuadratureSincConv1d(_QuadratureSincConv1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = super().forward(input)
        output = torch.abs(x)

        return output
