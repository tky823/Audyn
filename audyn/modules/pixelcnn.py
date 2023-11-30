import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = [
    "Conv2d",
    "CausalConv2d",
    "VerticalConv2d",
    "HorizontalConv2d",
]


class Conv2d(nn.Module):
    """2D convolution for PixelCNN.

    Args:
        capture_center (bool): If ``True``, center is captured by convolution (a.k.a Mask B
            in original paper). Otherwise, center is ignored (a.k.a Mask A).

    .. note::

        When ``kernel_size=(5, 5)``, ``capture_center=True``, convolution kernel is
        shown as follows:

        |0.1|0.4|0.3|0.1|0.8|
        |0.3|0.6|0.3|0.2|0.6|
        |0.5|0.4|0.7|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|

        where ``0.0`` means padding value.

        When ``kernel_size=(5, 5)``, ``capture_center=False``, convolution kernel is
        shown as follows:

        |0.1|0.4|0.3|0.1|0.8|
        |0.3|0.6|0.3|0.2|0.6|
        |0.5|0.4|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        padding: Union[_size_2_t, str] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        capture_center: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        kernel_size = _pair(kernel_size)
        padding = _pair(padding)
        dilation = _pair(dilation)

        # validate kernel size
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("kernel_size is expected to be odd, but even number is given.")

        num_pixels = (kernel_size[0] // 2) * kernel_size[1] + kernel_size[1] // 2 + capture_center

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.capture_center = capture_center
        self.num_pixels = num_pixels

        weight = torch.empty((out_channels, in_channels // groups, num_pixels), **factory_kwargs)
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"

        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        s += ", capture_center={capture_center}"
        return s.format(**self.__dict__)

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ):
        assert self.padding_mode == "zeros", "Only 'zeros' is supported as padding_mode."

        return F.conv2d(input, weight, bias, 1, self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv2d in PixelCNN.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height', width').

        """
        in_channels, out_channels = self.in_channels, self.out_channels
        kernel_size = self.kernel_size
        groups = self.groups
        num_pixels = self.num_pixels
        num_total_pixels = kernel_size[0] * kernel_size[1]
        weight = F.pad(self.weight, (0, num_total_pixels - num_pixels))
        weight = weight.view(out_channels, in_channels // groups, kernel_size[0], kernel_size[1])

        return self._conv_forward(input, weight, self.bias)


class CausalConv2d(Conv2d):
    """Alias of Conv2d."""

    pass


class VerticalConv2d(nn.Conv2d):
    """Vertical 2D convolution.

    .. note::

        When ``kernel_size=(5, 5)``, ``capture_center=True``, convolution kernel is
        shown as follows:

        |0.6|0.4|0.2|0.5|0.3|
        |0.1|0.4|0.5|0.2|0.7|
        |0.3|0.2|0.2|0.5|0.1|
        |0.0|0.0|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|

        where ``0.0`` means padding value.

        When ``kernel_size=(5, 5)``, ``capture_center=False``, convolution kernel is
        shown as follows:

        |0.6|0.4|0.2|0.5|0.3|
        |0.1|0.4|0.5|0.2|0.7|
        |0.0|0.0|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|
        |0.0|0.0|0.0|0.0|0.0|

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        groups: int = 1,
        bias: bool = True,
        capture_center: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        kernel_height, kernel_width = _pair(kernel_size)

        # validate kernel size
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("kernel_size is expected to be odd, but even number is given.")

        if capture_center:
            kernel_size = (kernel_height // 2, kernel_width)
        else:
            kernel_size = (kernel_height // 2 + 1, kernel_width)

        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=groups,
            bias=bias,
            padding_mode="zeros",
            device=device,
            dtype=dtype,
        )

        self.capture_center = capture_center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        real_kernel_height, real_kernel_width = self.kernel_size
        padding_width = real_kernel_width - 1
        padding_left = real_kernel_width // 2
        padding_right = padding_width - padding_left

        # to avoid error caused by size 0
        x = F.pad(input, (padding_left, padding_right, real_kernel_height, -1))
        output = F.conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return output


class HorizontalConv2d(nn.Conv2d):
    """Horizontal 2D convolution.

    .. note::

        When ``kernel_size=5``, ``capture_center=True``, convolution kernel is
        shown as follows:

        |0.2|0.3|0.5|0.0|0.0|

        where ``0.0`` means padding value.

        When ``kernel_size=5``, ``capture_center=False``, convolution kernel is
        shown as follows:

        |0.2|0.3|0.0|0.0|0.0|

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
        bias: bool = True,
        capture_center: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        # validate kernel size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size is expected to be odd, but even number is given.")

        if capture_center:
            kernel_size = (1, kernel_size // 2 + 1)
        else:
            kernel_size = (1, kernel_size // 2)

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
            device=device,
            dtype=dtype,
        )

        self.capture_center = capture_center

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        capture_center = self.capture_center
        _, real_kernel_width = self.kernel_size

        if capture_center:
            padding_left = real_kernel_width - 1
            padding_right = 0
        else:
            padding_left = real_kernel_width
            padding_right = -1

        x = F.pad(input, (padding_left, padding_right))
        output = F.conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        return output
