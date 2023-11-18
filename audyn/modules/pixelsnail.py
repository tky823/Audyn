"""Modules of PixelSNAIL.

See https://arxiv.org/abs/1712.09763 for the details.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t

__all__ = ["Conv2d", "CausalConv2d"]


class Conv2d(nn.Conv2d):
    """Causal convolution for PixelSNAIL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        groups: int = 1,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv2d in PixelSNAIL.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height, width).

        .. note::

            When ``kernel_size`` is ``(2, 3)`` and ``input.size()`` is ``(*, *, 5, 6)``,
            the point represented as 'x' in the top figure depends on the receptive field
            represented as '*' in the bottom figure.

            output:
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|
                |-|-|-|-|x|-|
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|

            input:
                |-|-|-|-|-|-|
                |-|-|*|*|*|-|
                |-|-|*|*|*|-|
                |-|-|-|-|-|-|
                |-|-|-|-|-|-|

        """
        kernel_height, kernel_width = self.kernel_size

        x = F.pad(input, (kernel_width - 1, 0, kernel_height - 1, 0))

        return super().forward(x)


class CausalConv2d(Conv2d):
    """Alias of Conv2d."""

    pass
