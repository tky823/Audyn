from typing import List, Union

import torch
import torch.nn as nn
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 3,
        stride: Union[int, List[int]] = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_bins: int = 128,
    ) -> None:
        super().__init__()

        if isinstance(stride, int):
            stride = [stride] * 2
        else:
            assert len(stride) == 2, "Stride must be int or list of two ints."

        backbone = []

        for layer_index in range(num_layers):
            if layer_index == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels

            _out_channels = hidden_channels
            _stride = (2, stride[layer_index])

            block = ResidualBlock2d(_in_channels, _out_channels, kernel_size, stride=_stride)
            backbone.append(block)

        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(hidden_channels * n_bins // (2**num_layers), out_channels)
        self.dropout = nn.Dropout(p=dropout)

        self.n_bins = n_bins

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv2dSubsampling.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Subsampled feature of shape (batch_size, n_frames', out_channels).

        """
        n_bins = self.n_bins

        assert input.size(-2) == n_bins, (
            f"Shape of input should be (*, n_bins={n_bins}, n_frames)."
        )

        x = self.backbone(input)
        batch_size, _n_channels, _n_bins, _n_frames = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, _n_frames, _n_channels * _n_bins)
        x = self.proj(x)
        output = self.dropout(x)

        return output


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 2,
    ) -> None:
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        Kh, Kw = kernel_size
        Sh, Sw = stride
        padding = Kh - Sh, Kw - Sw

        self.conv2d1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm2d1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2d2 = nn.BatchNorm2d(out_channels)

        if out_channels != in_channels or stride != (1, 1):
            self.conv2d3 = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, stride=stride
            )
            self.norm2d3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv2d3 = nn.Identity()
            self.norm2d3 = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.conv2d1(input)
        x = self.norm2d1(x)
        x = self.relu1(x)
        x = self.conv2d2(x)
        x = self.norm2d2(x)

        x_residual = self.conv2d3(input)
        x_residual = self.norm2d3(x_residual)

        x = x + x_residual
        output = self.relu2(x)

        return output
