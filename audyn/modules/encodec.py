import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from .transformer import get_activation

__all__ = ["EncoderBlock", "DecoderBlock", "ResidualUnit1d"]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class EncoderBlock(nn.Module):
    """Encoder block of EnCodec."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        dilation_rate: _size_1_t = 1,
        padding_mode: str = "reflect",
        num_layers: int = 1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        if num_layers == 1:
            assert dilation_rate == 1, "When num_layers=1, dilation_rate > 1 is unavailable."

        backbone = []

        for layer_idx in range(num_layers):
            dilation = dilation_rate**layer_idx
            unit = ResidualUnit1d(
                in_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding_mode=padding_mode,
                activation=activation,
                weight_regularization=weight_regularization,
                is_causal=is_causal,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)

        stride = _single(stride)
        (stride_out,) = stride
        kernel_size_out = 2 * stride_out

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.nonlinear1d = activation
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=stride,
        )

        self.kernel_size_out = _single(kernel_size_out)
        self.stride = stride
        self.padding_mode = padding_mode
        self.weight_regularization = weight_regularization
        self.is_causal = is_causal

        self.registered_weight_norms = set()

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.registered_weight_norms.add("backbone")
            self.weight_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.backbone(input)
        x = self.nonlinear1d(x)
        x = _pad1d(
            x,
            kernel_size=self.kernel_size_out,
            stride=self.stride,
            mode=self.padding_mode,
            is_causal=self.is_causal,
        )
        output = self.conv1d(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            for unit in self.backbone:
                unit: ResidualUnit1d
                unit.weight_norm_()

            self.registered_weight_norms.add("backbone")

        self.conv1d = weight_norm_fn(self.conv1d)
        self.registered_weight_norms.add("conv1d")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        for unit in self.backbone:
            unit: ResidualUnit1d
            unit.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.conv1d = remove_weight_norm_fn(self.conv1d, *remove_weight_norm_args)
        self.registered_weight_norms.remove("conv1d")


class DecoderBlock(nn.Module):
    """Decoder block of EnCodec."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        dilation_rate: _size_1_t = 1,
        padding_mode: str = "reflect",
        num_layers: int = 1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        if num_layers == 1:
            assert dilation_rate == 1, "When num_layers=1, dilation_rate > 1 is unavailable."

        stride = _single(stride)
        (stride_in,) = stride
        kernel_size_in = 2 * stride_in

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.nonlinear1d = activation
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
                padding_mode=padding_mode,
                activation=activation,
                weight_regularization=weight_regularization,
                is_causal=is_causal,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)

        self.kernel_size_in = _single(kernel_size_in)
        self.stride = stride
        self.padding_mode = padding_mode
        self.weight_regularization = weight_regularization
        self.is_causal = is_causal

        self.registered_weight_norms = set()

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.registered_weight_norms.add("backbone")
            self.weight_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size_in,) = self.kernel_size_in
        (stride,) = self.stride
        padding = kernel_size_in - stride

        if self.is_causal:
            padding_left = 0
            padding_right = padding
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = self.nonlinear1d(input)
        x = self.conv1d(x)
        x = F.pad(x, (-padding_left, -padding_right))
        output = self.backbone(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            for unit in self.backbone:
                unit: ResidualUnit1d
                unit.weight_norm_()

            self.registered_weight_norms.add("backbone")

        self.conv1d = weight_norm_fn(self.conv1d)
        self.registered_weight_norms.add("conv1d")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        for unit in self.backbone:
            unit: ResidualUnit1d
            unit.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.conv1d = remove_weight_norm_fn(self.conv1d, *remove_weight_norm_args)
        self.registered_weight_norms.remove("conv1d")


class ResidualUnit1d(nn.Module):
    """ResidualUnit used in Encodec.

    Args:
        num_features (int): Number of channels.
        kernel_size (_size_1_t): Kernel size of first convolution.
        dilation (_size_1_t): Dilation of first convolution. Default: ``1``.
        is_causal (bool): If ``True``, causality is guaranteed.
        use_shortcut (bool): If ``True``, use learned pointwise convolution for shortcut.
            Default: ``True``.

    """

    def __init__(
        self,
        num_features: int,
        kernel_size: _size_1_t,
        dilation: _size_1_t = 1,
        padding_mode: str = "reflect",
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
        use_shortcut: bool = True,
    ) -> None:
        super().__init__()

        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        assert kernel_size[0] % 2 == 1, "kernel_size should be odd number."

        if isinstance(activation, str):
            activation_in = get_activation(activation)
            activation_out = get_activation(activation)
        else:
            activation_in = activation
            activation_out = activation

        self.nonlinear1d_in = activation_in
        self.conv1d_in = nn.Conv1d(
            num_features, num_features // 2, kernel_size=kernel_size, dilation=dilation
        )
        self.nonlinear1d_out = activation_out
        self.conv1d_out = nn.Conv1d(num_features // 2, num_features, kernel_size=1)

        if use_shortcut:
            self.shortcut = nn.Conv1d(num_features, num_features, kernel_size=1)
        else:
            self.shortcut = None

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.weight_regularization = weight_regularization
        self.is_causal = is_causal

        self.registered_weight_norms = set()
        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        residual = input

        x = self.nonlinear1d_in(input)
        x = _pad1d(
            x, kernel_size=self.kernel_size, mode=self.padding_mode, is_causal=self.is_causal
        )
        x = self.conv1d_in(x)
        x = self.nonlinear1d_out(x)
        x = _pad1d(x, kernel_size=1, mode=self.padding_mode, is_causal=self.is_causal)
        x = self.conv1d_out(x)

        if self.shortcut is not None:
            output = x + self.shortcut(residual)
        else:
            output = x

        return output

    def _reset_parameters(self) -> None:
        weight_regularization = self.weight_regularization

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.weight_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv1d_in = weight_norm_fn(self.conv1d_in)
        self.conv1d_out = weight_norm_fn(self.conv1d_out)

        self.registered_weight_norms.add("conv1d_in")
        self.registered_weight_norms.add("conv1d_out")

        if self.shortcut is not None:
            self.shortcut = weight_norm_fn(self.shortcut)
            self.registered_weight_norms.add("shortcut")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv1d_in = remove_weight_norm_fn(self.conv1d_in, *remove_weight_norm_args)
        self.conv1d_out = remove_weight_norm_fn(self.conv1d_out, *remove_weight_norm_args)

        self.registered_weight_norms.remove("conv1d_in")
        self.registered_weight_norms.remove("conv1d_out")

        if self.shortcut is not None:
            self.shortcut = remove_weight_norm_fn(self.shortcut, *remove_weight_norm_args)
            self.registered_weight_norms.remove("shortcut")


def _pad1d(
    input: torch.Tensor,
    kernel_size: _size_1_t,
    stride: _size_1_t = 1,
    dilation: _size_1_t = 1,
    mode: str = "reflect",
    is_causal: bool = True,
) -> torch.Tensor:
    (kernel_size,) = _single(kernel_size)
    (stride,) = _single(stride)
    (dilation,) = _single(dilation)

    padding = (kernel_size - 1) * dilation + 1 - stride

    n_frames = math.ceil((input.size(-1) - kernel_size + padding) / stride)
    length = n_frames * stride + kernel_size - padding
    extra_padding = length - input.size(-1)

    if is_causal:
        padding_left = padding
        padding_right = 0
    else:
        padding_left = padding // 2
        padding_right = padding - padding_left

    output = F.pad(input, (padding_left, padding_right + extra_padding), mode=mode)

    return output
