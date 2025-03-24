import torch
import torch.nn as nn
from torch.nn.common_types import _size_2_t


class Encoder(nn.Module):
    """Encoder for VQVAE using MNIST."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 4,
        stride: _size_2_t = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.conv_block = StackedConvBlock(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            num_layers=num_layers,
        )
        self.residual_block = StackedResidualBlock(hidden_channels)
        self.bottleneck_conv2d = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv_block(input)
        x = self.residual_block(x)
        output = self.bottleneck_conv2d(x)

        return output


class Decoder(nn.Module):
    """Decoder for VQVAE using MNIST."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 4,
        stride: _size_2_t = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        self.bottleneck_conv2d = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
        )
        self.residual_block = StackedResidualBlock(hidden_channels)
        self.conv_block = StackedConvTransposeBlock(
            out_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            num_layers=num_layers,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.bottleneck_conv2d(input)
        x = self.residual_block(x)
        x = self.conv_block(x)
        output = self.sigmoid(x)

        return output


class StackedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 4,
        stride: _size_2_t = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        net = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels

            block = nn.Conv2d(
                _in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )
            net.append(block)

            if layer_idx != num_layers - 1:
                net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.net(input)

        return output


class StackedConvTransposeBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 4,
        stride: _size_2_t = 2,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        net = []

        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                _out_channels = out_channels
            else:
                _out_channels = hidden_channels

            block = nn.ConvTranspose2d(
                hidden_channels,
                _out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - stride) // 2,
            )
            net.append(block)

            if layer_idx != num_layers - 1:
                net.append(nn.ReLU())

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.net(input)

        return output


class StackedResidualBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        kernel_size: int = 3,
        num_layers: int = 2,
    ) -> None:
        super().__init__()

        net = []

        for _ in range(num_layers):
            block = ResidualBlock(
                num_features, num_features, num_features, kernel_size=kernel_size
            )
            net.append(block)

        self.net = nn.Sequential(*net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = self.net(input)

        return output


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()

        self.relu_in = nn.ReLU()
        self.conv2d_in = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.relu_out = nn.ReLU()
        self.conv2d_out = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.relu_in(input)
        x = self.conv2d_in(x)
        x = self.relu_out(x)
        x = self.conv2d_out(x)
        output = x + input

        return output
