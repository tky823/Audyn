from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from ..modules.convnext import StackedConvNeXtBlock1d, _get_normalization

__all__ = [
    "WaveNeXtVocoder",
]


class WaveNeXtVocoder(nn.Module):
    """WaveNeXt vocoder.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels.
        bottleneck_channels (int): Number of bottleneck channels in ConvNeXt blocks.
        kernel_size (list or _size_1_t): List of kernel sizes. Nth item corresponds to number of
            kernel size in nth block.
        norm (str): Layer normalization type. Only ``layer_norm`` is supported.
        activation (str, nn.Module, or callable): Activation module.
        pre_kernel_size (_size_1_t): Kernel size of pre transposed convolution. Default: ``7``.
        post_kernel_size (_size_1_t): Kernel size of post transposed convolution. Default: ``256``.
        post_stride (_size_1_t, optional): Stride of post transposed convolution. If ``None``,
            ``post_kernel_size`` is used.
        num_blocks (int): Number of ``ConvNeXtBlock1d``. Default: ``8``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        bottleneck_channels: int,
        kernel_size: Union[List[_size_1_t], _size_1_t],
        norm: str = "layer_norm",
        activation: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "gelu",
        pre_kernel_size: _size_1_t = 7,
        post_kernel_size: _size_1_t = 256,
        post_stride: Optional[_size_1_t] = None,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()

        (pre_kernel_size,) = _single(pre_kernel_size)

        if post_stride is None:
            post_stride = post_kernel_size

        if type(kernel_size) in [int, tuple]:
            kernel_size = [kernel_size] * num_blocks
        else:
            assert len(kernel_size) == num_blocks

        self.pre_conv1d = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=pre_kernel_size, stride=1
        )
        self.pre_norm = _get_normalization(norm, hidden_channels)
        self.backbone = StackedConvNeXtBlock1d(
            hidden_channels,
            bottleneck_channels,
            kernel_size=kernel_size,
            norm=norm,
            activation=activation,
            num_blocks=num_blocks,
        )
        self.post_conv1d = nn.ConvTranspose1d(
            hidden_channels,
            out_channels,
            kernel_size=post_kernel_size,
            stride=post_stride,
        )

        self.pre_kernel_size = _single(pre_kernel_size)
        self.post_kernel_size = _single(post_kernel_size)
        self.post_stride = _single(post_stride)
        self.num_blocks = num_blocks

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of WaveNeXtVocoder.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape
                (batch_size, in_channels, num_frames).

        Returns:
            torch.Tensor: Waveform-like feature of shape (batch_size, out_channels, num_frames'),
                where num_frames' represents number of upsampled frames.

        """
        (pre_kernel_size,) = self.pre_kernel_size

        padding = (pre_kernel_size - 1) // 2
        x = F.pad(input, (padding, padding))
        x = self.pre_conv1d(x)
        x = x.transpose(-2, -1)
        x = self.pre_norm(x)
        x = x.transpose(-2, -1)
        x = self.backbone(x)
        # TODO: padding for overlapping
        output = self.post_conv1d(x)

        return output

    @property
    def up_scale(self) -> int:
        (_up_scale,) = self.post_stride

        return _up_scale
