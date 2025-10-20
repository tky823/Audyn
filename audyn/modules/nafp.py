from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t


class NeuralAudioFingerprinterBackbone(nn.Module):
    """Backbone for Neural Audio Fingerprinter.

    Args:
        in_channels (int): Number of input channels.
        num_features (int or list): Number of features for each layer.
        kernel_size (int or list): Kernel size(s).
        stride (int or list, optional): Stride(s). Default: ``2``.
        num_layers (int, optional): Number of layers. If not specified, it is inferred
            from num_features.

    """

    def __init__(
        self,
        in_channels: int,
        num_features: Union[int, List[int]],
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]] = 2,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(num_features, int):
            assert num_layers is not None, "num_layers must be specified if num_features is int."

            num_features = [num_features] * num_layers
        else:
            if num_layers is None:
                num_layers = len(num_features)
            else:
                assert len(num_features) == num_layers, (
                    "num_features and num_layers must have same length if both are specified."
                )

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * num_layers
        else:
            assert len(kernel_size) == num_layers, (
                "kernel_size and num_layers must have same length if both are specified."
            )

        if isinstance(stride, int):
            stride = [stride] * num_layers
        else:
            assert len(stride) == num_layers, (
                "stride and num_layers must have same length if both are specified."
            )

        backbone = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = num_features[layer_idx - 1]

            _out_channels = num_features[layer_idx]
            _kernel_size = kernel_size[layer_idx]
            _stride = stride[layer_idx]

            block = StackedConvBlock(
                _in_channels,
                _out_channels,
                kernel_size=_kernel_size,
                stride=_stride,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        self.in_channels = in_channels

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of NeuralAudioFingerprinterBackbone.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (*, n_bins, n_frames).

        Returns:
            torch.Tensor: Downsampled feature of shape (*,).

        """
        in_channels = self.in_channels

        if input.dim() == 3:
            assert in_channels == 1

            x = input.unsqueeze(dim=-3)
        elif input.dim() == 4:
            x = input
        else:
            raise ValueError(
                "Input must be of shape (batch_size, n_bins, n_frames) or "
                "(batch_size, in_channels, n_bins, n_frames)."
            )

        for block in self.backbone:
            x = block(x)

        assert x.size()[-2:] == (1, 1), "Output should be of shape (*, 1, 1)."

        output = x.mean(dim=(-2, -1))

        return output


class NeuralAudioFingerprinterProjection(nn.Module):
    """Projection module for Neural Audio Fingerprinter."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int) -> None:
        super().__init__()

        assert in_channels % out_channels == 0, "in_channels must be divisible by embedding_dim."

        split_size = in_channels // out_channels

        self.split_size = split_size
        self.conv2d_in = nn.Conv2d(
            in_channels,
            out_channels * hidden_channels,
            kernel_size=1,
            stride=1,
            groups=out_channels,
        )
        self.activation = nn.ELU()
        self.conv2d_out = nn.Conv2d(
            out_channels * hidden_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            groups=out_channels,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of NeuralAudioFingerprinterProjection.

        Args:
            input (torch.Tensor): Feature of shape (*, in_channels).

        Returns:
            torch.Tensor: Projected feature of shape (*, out_channels).

        """
        *shape, in_channels = input.size()
        x = input.view(*shape, in_channels, 1, 1)
        x = self.conv2d_in(x)
        x = self.activation(x)
        x = self.conv2d_out(x)
        x = x.mean(dim=(-2, -1))
        output = F.normalize(x, p=2, dim=-1)

        return output


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

        *_, n_bins, n_frames = input.size()

        padding = ((n_frames - 1) // stride) * stride + kernel_size - n_frames
        padding_left = padding // 2
        padding_right = padding - padding_left
        x = F.pad(input, (padding_left, padding_right))

        x = self.block1(x)

        padding = ((n_bins - 1) // stride) * stride + kernel_size - n_bins
        padding_top = padding // 2
        padding_bottom = padding - padding_top
        x = F.pad(x, (0, 0, padding_top, padding_bottom))

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
