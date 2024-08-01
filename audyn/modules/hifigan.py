from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _pair, _single

__all__ = [
    "MultiReceptiveFieldFusion",
    "ResidualBlock1d",
    "StackedConvBlock1d",
    "ConvBlock1d",
    "ConvBlock2d",
]


IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-receptive field fusion module.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (list or _size_1_t): List of kernel sizes. Nth item corresponds to number of
            kernel size in nth block.
        dilation (list): List of dilation factors. This value is given to ``ResidualBlock1d``.
        negative_slope (float): Negative slope in leaky relu.
        up_kernel_size (_size_1_t): Kernel size in transposed convolution.
        up_stride (_size_1_t): Stride in transposed convolution.
        stacked (bool): If ``stacked=True``, ``StackedConvBlock1d`` (ResBlock1) is used. Otherwise,
            ``ConvBlock1d`` (ResBlock2) is used. Default: ``True``.
        num_layers (int): Number of layers in each ``ResidualBlock1d``. Default: ``3``.
        num_blocks (int): Number of ``ResidualBlock1d``.
        weight_regularization (str, optional): Weight regularization method.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[List[_size_1_t], _size_1_t],
        dilation: Union[List[_size_1_t], _size_1_t] = 1,
        negative_slope: float = 0.1,
        up_kernel_size: _size_1_t = None,
        up_stride: _size_1_t = None,
        stacked: bool = True,
        num_layers: int = 3,
        num_blocks: int = 3,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        assert up_kernel_size is not None and up_stride is not None

        self.num_blocks = num_blocks
        self.up_kernel_size, self.up_stride = _single(up_kernel_size), _single(up_stride)

        if type(kernel_size) in [int, tuple]:
            kernel_size = [kernel_size] * num_blocks
        else:
            assert len(kernel_size) == num_blocks

        self.upsample = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=up_kernel_size,
            stride=up_stride,
        )

        backbone = []

        for block_idx in range(num_blocks):
            _kernel_size = kernel_size[block_idx]
            block = ResidualBlock1d(
                out_channels,
                kernel_size=_kernel_size,
                dilation=dilation,
                negative_slope=negative_slope,
                stacked=stacked,
                num_layers=num_layers,
                weight_regularization=weight_regularization,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        # registered_weight_norms and registered_spectral_norms manage normalization status
        self.registered_weight_norms = set()
        self.registered_spectral_norms = set()

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.registered_weight_norms.add("backbone")
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.registered_spectral_norms.add("backbone")
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.upsample = weight_norm_fn(self.upsample)
        self.registered_weight_norms.add("upsample")

        if "backbone" not in self.registered_weight_norms:
            for block in self.backbone:
                block: ResidualBlock1d
                block.weight_norm_()

            self.registered_weight_norms.add("backbone")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.upsample = remove_weight_norm_fn(self.upsample, *remove_weight_norm_args)
        self.registered_weight_norms.remove("upsample")

        for block in self.backbone:
            block: ResidualBlock1d
            block.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.upsample = spectral_norm_fn(self.upsample)
        self.registered_spectral_norms.add("upsample")

        if "backbone" not in self.registered_spectral_norms:
            for block in self.backbone:
                block.spectral_norm_()

            self.registered_spectral_norms.add("backbone")

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.upsample = remove_spectral_norm_fn(self.upsample, *remove_spectral_norm_args)
        self.registered_spectral_norms.remove("upsample")

        for block in self.backbone:
            block: ResidualBlock1d
            block.remove_spectral_norm_()

        self.registered_spectral_norms.remove("backbone")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (up_kernel_size,), (up_stride,) = self.up_kernel_size, self.up_stride
        num_blocks = self.num_blocks

        padding = (up_kernel_size - up_stride) // 2
        x = self.upsample(input)
        x = F.pad(x, (-padding, -padding))

        skip = 0

        for block_idx in range(num_blocks):
            x = self.backbone[block_idx](x)
            skip = x + skip

        # normalize by number of blocks
        output = skip / num_blocks

        return output


class ResidualBlock1d(nn.Module):
    """Residual block in generator of HiFi-GAN.

    Args:
        num_features (int): Number of features.
        kernel_size (list): Kernel sizes in convolutions. Nth item corresponds to kernel size
            in nth convolution.
        dilation (list): Dilation factors in convolutions. Nth item corresponds to dilation factor
            in nth convolution.
        negative_slope (float): Negative slope in leaky relu.
        stacked (bool): If ``stacked=True``, ``StackedConvBlock1d`` (ResBlock1) is used. Otherwise,
            ``ConvBlock1d`` (ResBlock2) is used. Default: ``True``.
        num_layers (int): Number of ``ConvBlock1d`` or ``StackedConvBlock1d``, which should be
            equal to length of ``kernel_size`` and ``dilation``. Default: ``3``.
        weight_regularization (str, optional): Weight regularization method.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        num_features: int,
        kernel_size: List[_size_1_t],
        dilation: List[_size_1_t] = 1,
        negative_slope: float = 0.1,
        stacked: bool = True,
        num_layers: int = 3,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        if type(kernel_size) in [int, tuple]:
            kernel_size = [kernel_size] * num_layers
        else:
            assert len(kernel_size) == num_layers

        if type(dilation) in [int, tuple]:
            dilation = [dilation] * num_layers
        else:
            assert len(dilation) == num_layers

        backbone = []

        for layer_idx in range(num_layers):
            _kernel_size = kernel_size[layer_idx]
            _dilation = dilation[layer_idx]

            if stacked:
                cls = StackedConvBlock1d
                args = (num_features, num_features, num_features)
            else:
                cls = ConvBlock1d
                args = (num_features, num_features)

            block = cls(
                *args,
                kernel_size=_kernel_size,
                stride=1,
                dilation=_dilation,
                negative_slope=negative_slope,
                weight_regularization=weight_regularization,
                nonlinear_first=True,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_layers = self.num_layers

        x = input

        for layer_idx in range(num_layers):
            x = self.backbone[layer_idx](x)

        output = x + input

        return output

    def weight_norm_(self) -> None:
        for block in self.backbone:
            block.weight_norm_()

    def remove_weight_norm_(self) -> None:
        for block in self.backbone:
            block.remove_weight_norm_()

    def spectral_norm_(self) -> None:
        for block in self.backbone:
            block.spectral_norm_()

    def remove_spectral_norm_(self) -> None:
        for block in self.backbone:
            block.remove_spectral_norm_()


class StackedConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        dilation: _size_1_t = 1,
        groups: int = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
        nonlinear_first: bool = False,
    ) -> None:
        super().__init__()

        self.conv1d_in = ConvBlock1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
            nonlinear_first=nonlinear_first,
        )
        self.conv1d_out = ConvBlock1d(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=1,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
            nonlinear_first=nonlinear_first,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Downsampled feature of shape
                (batch_size, out_channels, length // stride).

        """
        x = self.conv1d_in(input)
        output = self.conv1d_out(x)

        return output

    def weight_norm_(self) -> None:
        self.conv1d_in.weight_norm_()
        self.conv1d_out.weight_norm_()

    def remove_weight_norm_(self) -> None:
        self.conv1d_in.remove_weight_norm_()
        self.conv1d_out.remove_weight_norm_()

    def spectral_norm_(self) -> None:
        self.conv1d_in.spectral_norm_()
        self.conv1d_out.spectral_norm_()

    def remove_spectral_norm_(self) -> None:
        self.conv1d_in.remove_spectral_norm_()
        self.conv1d_out.remove_spectral_norm_()


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        dilation: _size_1_t = 1,
        groups: int = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
        nonlinear_first: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = _single(kernel_size)
        self.dilation = _single(dilation)
        self.nonlinear_first = nonlinear_first

        if nonlinear_first:
            if negative_slope == 1:
                raise ValueError(
                    "nonlinear_first=True is specified, "
                    "but invalid value of negative_slope=1 is given."
                )
            else:
                self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )

        if not nonlinear_first:
            if negative_slope == 1:
                self.leaky_relu = None
            else:
                self.leaky_relu = nn.LeakyReLU(negative_slope)

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Downsampled feature of shape
                (batch_size, out_channels, length // stride).

        """
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation

        nonlinear_first = self.nonlinear_first
        leaky_relu = self.leaky_relu

        if nonlinear_first and leaky_relu is not None:
            x = leaky_relu(input)
        else:
            x = input

        padding = (kernel_size * dilation - dilation) // 2
        x = F.pad(x, (padding, padding))
        x = self.conv1d(x)

        if not nonlinear_first and leaky_relu is not None:
            output = leaky_relu(x)
        else:
            output = x

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv1d = weight_norm_fn(self.conv1d)

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv1d = remove_weight_norm_fn(self.conv1d, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv1d = spectral_norm_fn(self.conv1d)

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv1d = remove_spectral_norm_fn(self.conv1d, *remove_spectral_norm_args)


class ConvBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        groups: int = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
        nonlinear_first: bool = False,
    ) -> None:
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.nonlinear_first = nonlinear_first

        if nonlinear_first:
            if negative_slope == 1:
                raise ValueError(
                    "nonlinear_first=True is specified, "
                    "but invalid value of negative_slope=1 is given."
                )
            else:
                self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
        )

        if not nonlinear_first:
            if negative_slope == 1:
                self.leaky_relu = None
            else:
                self.leaky_relu = nn.LeakyReLU(negative_slope)

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, length, period).

        Returns:
            torch.Tensor: Downsampled feature of shape
                (batch_size, out_channels, length // stride, period).

        """
        kh, kw = self.kernel_size

        nonlinear_first = self.nonlinear_first
        leaky_relu = self.leaky_relu

        if nonlinear_first and leaky_relu is not None:
            x = leaky_relu(input)
        else:
            x = input

        padding_height = (kh - 1) // 2
        padding_width = (kw - 1) // 2
        x = F.pad(x, (padding_width, padding_width, padding_height, padding_height))
        x = self.conv2d(x)

        if not nonlinear_first and leaky_relu is not None:
            output = leaky_relu(x)
        else:
            output = x

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d = weight_norm_fn(self.conv2d)

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d = remove_weight_norm_fn(self.conv2d, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d = spectral_norm_fn(self.conv2d)

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d = remove_spectral_norm_fn(self.conv2d, *remove_spectral_norm_args)
