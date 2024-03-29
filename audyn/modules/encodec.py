from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

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
            activation = _get_activation(activation)

        self.nonlinear1d = activation
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size_out,
            stride=stride,
        )

        self.kernel_size_out = _single(kernel_size_out)
        self.stride = stride
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
        (kernel_size_out,) = self.kernel_size_out
        (stride,) = self.stride
        padding = kernel_size_out - stride

        if self.is_causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))
        x = self.backbone(x)
        x = self.nonlinear1d(x)
        output = self.conv1d(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "backbone" not in self.registered_weight_norms:
            self.backbone.weight_norm_()
            self.registered_weight_norms.add("backbone")

        self.conv1d = weight_norm_fn(self.conv1d)
        self.registered_weight_norms.add("conv1d")


class DecoderBlock(nn.Module):
    """Decoder block of EnCodec."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t,
        dilation_rate: _size_1_t = 1,
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
            activation = _get_activation(activation)

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
                activation=activation,
                weight_regularization=weight_regularization,
                is_causal=is_causal,
            )
            backbone.append(unit)

        self.backbone = nn.Sequential(*backbone)

        self.kernel_size_in = _single(kernel_size_in)
        self.stride = stride
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
            self.backbone.weight_norm_()
            self.registered_weight_norms.add("backbone")

        self.conv1d = weight_norm_fn(self.conv1d)
        self.registered_weight_norms.add("conv1d")


class ResidualUnit1d(nn.Module):
    """ResidualUnit used in Encodec.

    Args:
        num_features (int): Number of channels.
        kernel_size (_size_1_t): Kernel size of first convolution.
        dilation (_size_1_t): Dilation of first convolution. Default: ``1``.
        is_causal (bool): If ``True``, causality is guaranteed.

    """

    def __init__(
        self,
        num_features: int,
        kernel_size: _size_1_t,
        dilation: _size_1_t = 1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.elu,
        weight_regularization: Optional[str] = "weight_norm",
        is_causal: bool = True,
    ) -> None:
        super().__init__()

        kernel_size = _single(kernel_size)
        dilation = _single(dilation)

        assert kernel_size[0] % 2 == 1, "kernel_size should be odd number."

        if isinstance(activation, str):
            activation_in = _get_activation(activation)
            activation_out = _get_activation(activation)
        else:
            activation_in = activation
            activation_out = activation

        self.nonlinear1d_in = activation_in
        self.conv1d_in = nn.Conv1d(
            num_features, num_features // 2, kernel_size=kernel_size, dilation=dilation
        )
        self.nonlinear1d_out = activation_out
        self.conv1d_out = nn.Conv1d(num_features // 2, num_features, kernel_size=1)

        self.kernel_size = kernel_size
        self.dilation = dilation
        self.weight_regularization = weight_regularization
        self.is_causal = is_causal

        self.registered_weight_norms = set()
        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation

        dilated_kernel_size = (kernel_size - 1) * dilation + 1

        if self.is_causal:
            padding = dilated_kernel_size // 2
            x = F.pad(input, (padding, padding))
        else:
            padding = dilated_kernel_size // 2
            x = F.pad(input, (2 * padding, 0))

        x = self.nonlinear1d_in(x)
        x = self.conv1d_in(x)
        x = self.nonlinear1d_out(x)
        x = self.conv1d_out(x)
        output = x + input

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


def _get_activation(activation: str) -> nn.Module:
    """Get activation module by str.

    Args:
        activation (str): Name of activation module.

    Returns:
        nn.Module: Activation module.

    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"activation should be relu/gelu/elu, not {activation}")
