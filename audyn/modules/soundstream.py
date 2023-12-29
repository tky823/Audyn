import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _pair, _single

__all__ = ["EncoderBlock", "DecoderBlock", "ResidualUnit1d", "ResidualUnit2d"]


class EncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        backbone = []

        for layer_idx in range(num_layers):
            dilation = dilation_rate**layer_idx
            unit = ResidualUnit1d(
                in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)

        stride = _single(stride)
        (stride_out,) = stride
        kernel_size_out = 2 * stride_out

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=stride,
        )

        self.kernel_size_out = _single(kernel_size_out)
        self.stride = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size_out,) = self.kernel_size_out
        (stride,) = self.stride
        padding = kernel_size_out - stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.backbone(x)
        output = self.conv1d(x)

        return output


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        dilation_rate: _size_1_t = 3,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        stride = _single(stride)
        (stride_in,) = stride
        kernel_size_in = 2 * stride_in

        self.conv1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_in,
            stride=stride,
        )

        backbone = []

        for layer_idx in range(num_layers):
            dilation = dilation_rate**layer_idx
            unit = ResidualUnit1d(
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)

        self.kernel_size_in = _single(kernel_size_in)
        self.stride = stride

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size_in,) = self.kernel_size_in
        (stride,) = self.stride
        padding = kernel_size_in - stride
        padding_left = padding // 2
        padding_right = padding - padding_left

        x = self.conv1d(input)
        x = F.pad(x, (-padding_left, -padding_right))
        output = self.backbone(x)

        return output


class ResidualUnit1d(nn.Module):
    def __init__(self, num_features: int, kernel_size: _size_1_t, dilation: _size_1_t = 1) -> None:
        super().__init__()

        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        assert kernel_size[0] % 2 == 1, "kernel_size should be odd number."

        self.conv1d_in = nn.Conv1d(
            num_features, num_features, kernel_size=kernel_size, dilation=dilation
        )
        self.nonlinear1d = nn.ELU()
        self.conv1d_out = nn.Conv1d(num_features, num_features, kernel_size=1)

        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation

        dilated_kernel_size = (kernel_size - 1) * dilation + 1
        padding = dilated_kernel_size // 2

        x = F.pad(input, (padding, padding))
        x = self.conv1d_in(x)
        x = self.nonlinear1d(x)
        x = self.conv1d_out(x)
        output = x + input

        return output


class ResidualUnit2d(nn.Module):
    """Residual unit to process 2D features used in discriminator of SoundStream."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        down_scale: _size_2_t,
    ) -> None:
        super().__init__()

        kernel_size = _pair(kernel_size)
        down_scale = _pair(down_scale)

        # not sure of implementation of bottleneck_conv2d
        self.bottleneck_conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=down_scale
        )

        kh, kw = down_scale
        kh, kw = kh + 2, kw + 2
        self.conv2d_in = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kh, kw),
            stride=1,
        )
        self.nonlinear2d = nn.ELU()
        self.conv2d_out = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=down_scale,
        )

        self.kernel_size = kernel_size
        self.down_scale = down_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResidualUnit2d.

        Args:
            input (torch.Tensor): Feature map of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, height', width'), where
                height' and width' are defined as ``height // down_scale[0]`` and
                ``width // down_scale[1]``, respectively.

        """
        kernel_size = self.kernel_size
        down_scale = self.down_scale

        x_in = self.bottleneck_conv2d(input)
        x = self._pad2d(input, kernel_size=(down_scale[0] + 2, down_scale[1] + 2))
        x = self.conv2d_in(x)
        x = self.nonlinear2d(x)
        x = self._pad2d(x, kernel_size=kernel_size)
        output = x_in + self.conv2d_out(x)

        return output

    def _pad2d(self, input: torch.Tensor, kernel_size: _size_2_t) -> torch.Tensor:
        kernel_size = _pair(kernel_size)

        kh, kw = kernel_size
        ph, pw = kh - 1, kw - 1
        padding_top = ph // 2
        padding_left = pw // 2
        padding_bottom = ph - padding_top
        padding_right = pw - padding_left

        output = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))

        return output