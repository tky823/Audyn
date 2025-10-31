import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t

__all__ = [
    "GLU",
    "GLU1d",
    "GLU2d",
]


class GLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.map = nn.Linear(in_channels, out_channels, bias=bias)
        self.map_gate = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of GLU.

        Args:
            input (torch.Tensor): Input of shape (*, in_channels).

        Returns:
            torch.Tensor: Output of shape (*, in_channels).

        """
        x_output = self.map(input)
        x_gate = self.map_gate(input)
        x_gate = F.sigmoid(x_gate)

        output = x_output * x_gate

        return output


class GLU1d(nn.Module):
    """Gated Linear Units for 1D inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.map = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.map_gate = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_output = self.map(input)
        x_gate = self.map_gate(input)
        x_gate = F.sigmoid(x_gate)

        output = x_output * x_gate

        return output


class GLU2d(nn.Module):
    """Gated Linear Units for 2D inputs."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.map = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.map_gate = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x_output = self.map(input)
        x_gate = self.map_gate(input)
        x_gate = F.sigmoid(x_gate)

        output = x_output * x_gate

        return output
