import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class StackedConvBlock(nn.Module):
    """Stacked convolution block for neural audio fingerprinting."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding_mode: str = "replicate",
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_mode = padding_mode

        self.block1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
        )

        self.block2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        kernel_size = self.kernel_size
        stride = self.stride

        padding = kernel_size - stride
        padding_head = padding // 2
        padding_tail = padding - padding_head

        x = F.pad(input, (padding_head, padding_tail))
        x = self.block1(x)
        x = F.pad(x, (0, 0, padding_head, padding_tail))
        output = self.block2(x)

        return output


class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
    ) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d(input)
        x = self.norm(x)
        output = self.activation(x)

        return output
