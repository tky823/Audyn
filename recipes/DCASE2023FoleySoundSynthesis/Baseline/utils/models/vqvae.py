"""Modules of VQVAE."""

import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class Encoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        res_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        num_blocks: int = 1,
        num_stacks: int = 2,
    ) -> None:
        super().__init__()

        in_channels = 1

        self.num_blocks = num_blocks
        self.kernel_size = kernel_size

        backbone = []

        for block_idx in range(num_blocks):
            block = EncoderBlock(
                in_channels,
                hidden_channels,
                res_channels,
                kernel_size=(block_idx + 1) * kernel_size,
                stride=stride,
                num_stacks=num_stacks,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)
        self.bottleneck_conv2d = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_blocks = self.num_blocks

        input = input.unsqueeze(dim=1)

        x = 0

        for block_idx in range(num_blocks):
            x = x + self.backbone[block_idx](input)

        output = self.bottleneck_conv2d(x)

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        num_stacks: int = 2,
    ) -> None:
        super().__init__()

        out_channels = 1

        self.bottleneck_conv2d = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)

        res_blocks = []

        for _ in range(num_stacks):
            res_blocks.append(
                ResidualBlock(
                    hidden_channels,
                    res_channels,
                )
            )

        self.res_blocks = nn.Sequential(*res_blocks)
        self.relu_1 = nn.ReLU()
        self.conv_transpose2d_1 = nn.ConvTranspose2d(
            hidden_channels,
            hidden_channels // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // stride - 1,
        )
        self.relu_2 = nn.ReLU()
        self.conv_transpose2d_2 = nn.ConvTranspose2d(
            hidden_channels // 2,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // stride - 1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck_conv2d(input)
        x = self.res_blocks(x)
        x = self.relu_1(x)
        x = self.conv_transpose2d_1(x)
        x = self.relu_2(x)
        x = self.conv_transpose2d_2(x)
        output = x.squeeze(dim=1)

        return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        res_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        num_stacks: int = 2,
    ) -> None:
        super().__init__()

        self.conv2d_1 = nn.Conv2d(
            in_channels,
            hidden_channels // 2,
            kernel_size,
            stride=stride,
            padding=kernel_size // stride - 1,
        )
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(
            hidden_channels // 2,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // stride - 1,
        )
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        res_blocks = []

        for _ in range(num_stacks):
            res_blocks.append(ResidualBlock(hidden_channels, res_channels))

        self.res_blocks = nn.Sequential(*res_blocks)
        self.relu_3 = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv2d_1(input)
        x = self.relu_1(x)
        x = self.conv2d_2(x)
        x = self.relu_2(x)
        x = self.conv2d_3(x)
        x = self.res_blocks(x)
        output = self.relu_3(x)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, num_features: int, hidden_channels: int) -> None:
        super().__init__()

        self.relu_1 = nn.ReLU()
        self.conv2d_1 = nn.Conv2d(num_features, hidden_channels, kernel_size=3, padding=1)
        self.relu_2 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(hidden_channels, num_features, kernel_size=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.relu_1(input)
        x = self.conv2d_1(x)
        x = self.relu_2(x)
        output = input + self.conv2d_2(x)

        return output
