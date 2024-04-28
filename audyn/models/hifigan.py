from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_1_t, _size_2_t
from torch.nn.modules.utils import _pair, _single

__all__ = [
    "Generator",
    "Discriminator",
    "HiFiGANGenerator",
    "HiFiGANDiscriminator",
    "MultiScaleDiscriminator",
    "MultiPeriodDiscriminator",
]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")
available_weight_regularizations = {"weight_norm", "spectral_norm"}


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: Union[List[_size_1_t], _size_1_t],
        dilation: Union[List[List[_size_1_t]], List[_size_1_t], _size_1_t] = 1,
        negative_slope: float = 0.1,
        up_kernel_size: _size_1_t = None,
        up_stride: _size_1_t = None,
        pre_kernel_size: _size_1_t = 7,
        post_kernel_size: _size_1_t = 7,
        stacked: bool = True,
        num_layers: int = 3,
        num_blocks: int = 3,
        num_stacks: int = 4,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.num_stacks = num_stacks
        self.pre_kernel_size = _single(pre_kernel_size)
        self.post_kernel_size = _single(post_kernel_size)
        self.up_stride = up_stride

        if type(dilation) in [int, tuple]:
            dilation = [dilation] * num_stacks
        elif type(dilation) is list:
            if type(dilation[0]) in [int, tuple]:
                dilation = [dilation] * num_stacks
            elif type(dilation) is not list:
                raise TypeError("Invalid type of dilation is given.")
        else:
            raise TypeError("Invalid type of dilation is given.")

        self.pre_conv1d = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=pre_kernel_size, stride=1
        )

        backbone = []

        for stack_idx in range(num_stacks):
            _in_channels = hidden_channels // 2**stack_idx
            _out_channels = hidden_channels // 2 ** (stack_idx + 1)

            stack = MultiReceptiveFieldFusion(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                dilation=dilation[stack_idx],
                negative_slope=negative_slope,
                up_kernel_size=up_kernel_size[stack_idx],
                up_stride=up_stride[stack_idx],
                stacked=stacked,
                num_layers=num_layers,
                num_blocks=num_blocks,
                weight_regularization=weight_regularization,
            )
            backbone.append(stack)

        assert hidden_channels // 2**num_stacks > 0, "Number of channels became 0."

        self.backbone = nn.ModuleList(backbone)

        self.post_conv1d = ConvBlock1d(
            hidden_channels // 2**num_stacks,
            out_channels,
            kernel_size=post_kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
            nonlinear_first=True,
        )
        self.tanh = nn.Tanh()

        # registered_weight_norms and registered_spectral_norms manage normalization status
        self.registered_weight_norms = set()
        self.registered_spectral_norms = set()

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.registered_weight_norms.add("backbone")
                self.registered_weight_norms.add("post_conv1d")
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.registered_spectral_norms.add("backbone")
                self.registered_spectral_norms.add("post_conv1d")
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

        self.pre_conv1d = weight_norm_fn(self.pre_conv1d)
        self.registered_weight_norms.add("pre_conv1d")

        if "backbone" not in self.registered_weight_norms:
            for stack in self.backbone:
                stack: MultiReceptiveFieldFusion
                stack.weight_norm_()

            self.registered_weight_norms.add("backbone")

        if "post_conv1d" not in self.registered_weight_norms:
            self.post_conv1d.weight_norm_()
            self.registered_weight_norms.add("post_conv1d")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.pre_conv1d = remove_weight_norm_fn(self.pre_conv1d, *remove_weight_norm_args)
        self.registered_weight_norms.remove("pre_conv1d")

        for stack in self.backbone:
            stack: MultiReceptiveFieldFusion
            stack.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.post_conv1d.remove_weight_norm_()
        self.registered_weight_norms.remove("post_conv1d")

    def spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.pre_conv1d = spectral_norm_fn(self.pre_conv1d)
        self.registered_spectral_norms.add("pre_conv1d")

        if "backbone" not in self.registered_spectral_norms:
            for stack in self.backbone:
                stack: MultiReceptiveFieldFusion
                stack.spectral_norm_()

            self.registered_spectral_norms.add("backbone")

        if "post_conv1d" not in self.registered_weight_norms:
            self.post_conv1d.spectral_norm_()
            self.registered_spectral_norms.add("post_conv1d")

    def remove_spectral_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.pre_conv1d = remove_spectral_norm_fn(self.pre_conv1d, *remove_spectral_norm_args)
        self.registered_spectral_norms.remove("pre_conv1d")

        for stack in self.backbone:
            stack: MultiReceptiveFieldFusion
            stack.remove_spectral_norm_()

        self.registered_spectral_norms.remove("backbone")

        self.post_conv1d.remove_spectral_norm_()
        self.registered_spectral_norms.remove("post_conv1d")

    @classmethod
    def build_from_default_config(cls, variation: Union[str, int]) -> "Generator":
        if type(variation) is int:
            variation = "v" + str(variation)

        if variation.lower() == "v1":
            hidden_channels = 512
            kernel_size = [3, 7, 11]
            dilation = [1, 3, 5]
            up_kernel_size, up_stride = [16, 16, 4, 4], [8, 8, 2, 2]
            stacked = True
            num_stacks, num_layers = len(up_kernel_size), 3
        elif variation.lower() == "v2":
            hidden_channels = 128
            kernel_size = [3, 7, 11]
            dilation = [1, 3, 5]
            up_kernel_size, up_stride = [16, 16, 4, 4], [8, 8, 2, 2]
            stacked = True
            num_stacks, num_layers = len(up_kernel_size), 3
        elif variation.lower() == "v3":
            hidden_channels = 256
            kernel_size = [3, 5, 7]
            dilation = [[1, 2], [2, 6], [6, 12]]
            up_kernel_size, up_stride = [16, 16, 8], [8, 8, 4]
            stacked = False
            num_stacks, num_layers = len(up_kernel_size), 2

        in_channels, out_channels = 80, 1
        pre_kernel_size, post_kernel_size = 7, 7
        num_blocks = 3
        weight_regularization = "weight_norm"

        model = cls(
            in_channels,
            out_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            up_kernel_size=up_kernel_size,
            up_stride=up_stride,
            pre_kernel_size=pre_kernel_size,
            post_kernel_size=post_kernel_size,
            stacked=stacked,
            num_layers=num_layers,
            num_blocks=num_blocks,
            num_stacks=num_stacks,
            weight_regularization=weight_regularization,
        )

        return model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_stacks = self.num_stacks
        (pre_kernel_size,) = self.pre_kernel_size

        padding = (pre_kernel_size - 1) // 2
        x = F.pad(input, (padding, padding))
        x = self.pre_conv1d(x)

        for stack_idx in range(num_stacks):
            x = self.backbone[stack_idx](x)

        x = self.post_conv1d(x)
        output = self.tanh(x)

        return output


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


class Discriminator(nn.Module):
    def __init__(
        self,
        period_discriminator: "MultiPeriodDiscriminator",
        scale_discriminator: "MultiScaleDiscriminator",
    ) -> None:
        super().__init__()

        self.period_discriminator = period_discriminator
        self.scale_discriminator = scale_discriminator

    def forward(self, input: torch.Tensor) -> Tuple[Any, Any]:
        """Forward pass of multi-scale discriminator.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length),
                where in_channels corresponds to ``num_features[0]``.

        Returns:
            tuple: Tuple containing

                - any: Outputs from period discriminator. If ``MultiPeriodDiscriminator`` \
                    is used, type is tuple.
                - any: Outputs from scale discriminator. If ``MultiScaleDiscriminator`` \
                    is used, type is tuple.

        """
        period_output, period_feature_map = self.period_discriminator(input)
        scale_output, scale_feature_map = self.scale_discriminator(input)

        return (period_output, scale_output), (period_feature_map, scale_feature_map)


class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator.

    Args:
        num_features (list): Number of features in convolution. This value is given to
            each sub-discriminator.
        kernel_size (list): Kernel sizes in convolution. This value is given to
            each sub-discriminator.
        stride (list or _size_1_t): Strides in convolution. This value is given to
            each sub-discriminator.
        dilation (list or _size_1_t): Dilation factor in convolution. This value is given to
            each sub-discriminator.
        groups (list or int): List of groupds in convolution. This value is given to
            each sub-discriminator.
        negative_slope (float): Negative slope in leaky relu.
        pool_kernel_size (_size_1_t): Kernel size in pooling layer.
        pool_stride (_size_1_t): Stride in pooling layer.
        weight_regularization (list, optional): List of weight regularization methods,
            whose length corresponds to number of sub-discriminators.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        num_features: List[int],
        kernel_size: List[_size_1_t],
        stride: Union[List[_size_1_t], _size_1_t] = 1,
        dilation: Union[List[_size_1_t], _size_1_t] = 1,
        groups: Union[List[int], int] = 1,
        negative_slope: float = 0.1,
        pool_kernel_size: _size_1_t = 4,
        pool_stride: _size_1_t = 2,
        weight_regularization: Optional[List[Optional[str]]] = None,
    ):
        super().__init__()

        if weight_regularization is None:
            weight_regularization = [None, None, None]

        pool_kernel_size = _single(pool_kernel_size)
        pool_stride = _single(pool_stride)

        assert type(weight_regularization) is list

        num_discriminators = len(weight_regularization)

        discriminator = []
        pool = []

        for discriminator_idx in range(num_discriminators):
            if discriminator_idx != 0:
                padding = pool_kernel_size[0] - pool_stride[0]
                pool.append(
                    nn.AvgPool1d(
                        pool_kernel_size,
                        stride=pool_stride,
                        padding=padding,
                    ),
                )

            scale_discriminator = ScaleDiscriminator(
                num_features,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                groups=groups,
                negative_slope=negative_slope,
                weight_regularization=weight_regularization[discriminator_idx],
            )
            discriminator.append(scale_discriminator)

        self.discriminator = nn.ModuleList(discriminator)
        self.pool = nn.ModuleList(pool)

        self.num_discriminators = num_discriminators
        self.weight_regularization = weight_regularization

    def weight_regularization_(self) -> None:
        for discriminator_idx in range(self.num_discriminators):
            if self.weight_regularization[discriminator_idx] == "weight_norm":
                self.discriminator[discriminator_idx].weight_norm_()
            elif self.weight_regularization[discriminator_idx] == "spectral_norm":
                self.discriminator[discriminator_idx].spectral_norm_()

    def remove_weight_regularization_(self) -> None:
        for discriminator_idx in range(self.num_discriminators):
            if self.weight_regularization[discriminator_idx] == "weight_norm":
                self.discriminator[discriminator_idx].remove_weight_norm_()
            elif self.weight_regularization[discriminator_idx] == "spectral_norm":
                self.discriminator[discriminator_idx].remove_spectral_norm_()

    @classmethod
    def build_from_default_config(cls) -> "MultiScaleDiscriminator":
        """Build multi-scale discriminator from default config.

        Returns:
            MultiScaleDiscriminator: Multi-scale discriminator by default parameters.

        """
        num_features = [1, 128, 128, 256, 512, 1024, 1024, 1024]
        kernel_size = [15, 41, 41, 41, 41, 41, 5, 3]
        stride = [1, 2, 2, 4, 4, 1, 1, 1]
        dilation = [1, 1, 1, 1, 1, 1, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1, 1]
        negative_slope = 0.1
        pool_kernel_size = 4
        pool_stride = 2
        weight_regularization = ["spectral_norm", "weight_norm", "weight_norm"]

        discriminator = cls(
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            negative_slope=negative_slope,
            pool_kernel_size=pool_kernel_size,
            pool_stride=pool_stride,
            weight_regularization=weight_regularization,
        )

        return discriminator

    def forward(self, input: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass of multi-scale discriminator.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length),
                where in_channels corresponds to ``num_features[0]``.

        Returns:
            tuple: Tuple of tensors containing

                - list: List of downsampled feature of shape (batch_size, 1, length // down_scale).
                - list: Nested list torch.Tensor. Nth item is output of nth discriminator.

        """

        output = []
        feature_map = []

        for discriminator_idx, discriminator in enumerate(self.discriminator):
            if discriminator_idx == 0:
                x = input
            else:
                x = self.pool[discriminator_idx - 1](x)

            _output, _feature_map = discriminator(x)

            output.append(_output)
            feature_map.append(_feature_map)

        return output, feature_map


class MultiPeriodDiscriminator(nn.Module):
    """Multi-period discriminator.

    Args:
        period (list): List of periods (int), whose length corresponds to
            number of sub-discriminators.
        num_features (list): Number of features in convolution. This value is given to
            each sub-discriminator.
        kernel_size (list): Kernel sizes in convolution. This value is given to
            each sub-discriminator.
        stride (list or _size_1_t): Strides in convolution. This value is given to
            each sub-discriminator.
        groups (list or int): List of groupds in convolution. This value is given to
            each sub-discriminator.
        negative_slope (float): Negative slope in leaky relu.
        weight_regularization (list, optional): List of weight regularization methods,
            whose length corresponds to number of sub-discriminators.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        period: List[int],
        num_features: List[int],
        kernel_size: List[int],
        stride: Union[List[int], _size_1_t] = 1,
        groups: Union[List[int], int] = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[List[Optional[str]]] = None,
    ) -> None:
        super().__init__()

        num_discriminators = len(period)

        if weight_regularization is None:
            weight_regularization = [None] * num_discriminators

        assert type(weight_regularization) is list
        assert len(weight_regularization) == num_discriminators

        discriminator = []

        for discriminator_idx in range(num_discriminators):
            period_discriminator = PeriodDiscriminator(
                period[discriminator_idx],
                num_features,
                kernel_size=kernel_size,
                stride=stride,
                groups=groups,
                negative_slope=negative_slope,
                weight_regularization=weight_regularization[discriminator_idx],
            )
            discriminator.append(period_discriminator)

        self.discriminator = nn.ModuleList(discriminator)

        self.num_discriminators = num_discriminators
        self.weight_regularization = weight_regularization

    def weight_regularization_(self) -> None:
        for discriminator_idx in range(self.num_discriminators):
            if self.weight_regularization[discriminator_idx] == "weight_norm":
                self.discriminator[discriminator_idx].weight_norm_()
            elif self.weight_regularization[discriminator_idx] == "spectral_norm":
                self.discriminator[discriminator_idx].spectral_norm_()

    def remove_weight_regularization_(self) -> None:
        for discriminator_idx in range(self.num_discriminators):
            if self.weight_regularization[discriminator_idx] == "weight_norm":
                self.discriminator[discriminator_idx].remove_weight_norm_()
            elif self.weight_regularization[discriminator_idx] == "spectral_norm":
                self.discriminator[discriminator_idx].remove_spectral_norm_()

    @classmethod
    def build_from_default_config(cls) -> "MultiPeriodDiscriminator":
        """Build multi-period discriminator from default config.

        Returns:
            MultiPeriodDiscriminator: Multi-period discriminator by default parameters.

        """
        period = [2, 3, 5, 7, 11]
        num_features = [1, 32, 128, 512, 1024, 1024]
        kernel_size = [5, 5, 5, 5, 5, 3]
        stride = [3, 3, 3, 3, 1, 1]
        groups = [1, 1, 1, 1, 1, 1]
        negative_slope = 0.1
        weight_regularization = [
            "weight_norm",
            "weight_norm",
            "weight_norm",
            "weight_norm",
            "weight_norm",
        ]

        discriminator = cls(
            period,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
        )

        return discriminator

    def forward(self, input: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        """Forward pass of multi-period discriminator.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length),
                where in_channels corresponds to ``num_features[0]``.

        Returns:
            tuple: Tuple of tensors containing

                - list: List of downsampled feature of shape (batch_size, 1, length // down_scale).
                - list: Nested list torch.Tensor. Nth item is output of nth discriminator.

        """
        output = []
        feature_map = []

        for discriminator in self.discriminator:
            _output, _feature_map = discriminator(input)

            output.append(_output)
            feature_map.append(_feature_map)

        return output, feature_map


class ScaleDiscriminator(nn.Module):
    """Scale discriminator.

    Args:
        num_features (list): Number of features in convolution.
        kernel_size (list): Kernel sizes in convolution.
        stride (list or _size_1_t): Strides in convolution.
        dilation (list or _size_1_t): Dilation factor in convolution.
        groups (list or int): Number of groupds in convolution.
        negative_slope (float): Negative slope in leaky relu.
        weight_regularization (list, optional): Weight regularization method.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        num_features: List[int],
        kernel_size: List[_size_1_t],
        stride: Union[List[_size_1_t], _size_1_t] = 1,
        dilation: Union[List[_size_1_t], _size_1_t] = 1,
        groups: Union[List[int], int] = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        num_layers = len(num_features)

        if type(kernel_size) in [int, tuple]:
            kernel_size = [kernel_size] * num_layers
        else:
            assert len(kernel_size) == num_layers

        if type(stride) in [int, tuple]:
            stride = [stride] * num_layers
        else:
            assert len(stride) == num_layers

        if type(dilation) in [int, tuple]:
            dilation = [dilation] * num_layers
        else:
            assert len(dilation) == num_layers

        if type(groups) in [int, tuple]:
            groups = [groups] * num_layers
        else:
            assert len(groups) == num_layers

        if weight_regularization is not None:
            assert weight_regularization in available_weight_regularizations

        net = []

        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                # Last layer does not use leaky relu.
                out_channels = 1
                _negative_slope = 1
            else:
                out_channels = num_features[layer_idx + 1]
                _negative_slope = negative_slope

            in_channels = num_features[layer_idx]
            _kernel_size = kernel_size[layer_idx]
            _stride = stride[layer_idx]
            _dilation = dilation[layer_idx]
            _groups = groups[layer_idx]

            net.append(
                ConvBlock1d(
                    in_channels,
                    out_channels,
                    kernel_size=_kernel_size,
                    stride=_stride,
                    dilation=_dilation,
                    groups=_groups,
                    negative_slope=_negative_slope,
                    weight_regularization=weight_regularization,
                    nonlinear_first=False,
                )
            )

        self.net = nn.ModuleList(net)

        self.num_layers = num_layers

    def weight_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].weight_norm_()

    def remove_weight_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].remove_weight_norm_()

    def spectral_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].spectral_norm_()

    def remove_spectral_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].remove_spectral_norm_()

    @classmethod
    def build_from_default_config(
        cls, weight_regularization: Optional[str] = "weight_norm"
    ) -> "ScaleDiscriminator":
        """Build scale discriminator from default config.

        Returns:
            ScaleDiscriminator: Scale discriminator by default parameters.

        """
        num_features = [1, 128, 128, 256, 512, 1024, 1024, 1024]
        kernel_size = [15, 41, 41, 41, 41, 41, 5, 3]
        stride = [1, 2, 2, 4, 4, 1, 1, 1]
        dilation = [1, 1, 1, 1, 1, 1, 1, 1]
        groups = [1, 4, 16, 16, 16, 16, 1, 1]
        negative_slope = 0.1

        discriminator = cls(
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
        )

        return discriminator

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of scale discriminator.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length),
                where in_channels should be equal to ``num_features[0]``.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Downsampled feature of shape (batch_size, 1, length // down_scale).
                - list: List of torch.Tensor. Nth item is output of nth layer.

        """
        x = input
        feature_map = []

        for layer_idx in range(self.num_layers):
            x = self.net[layer_idx](x)
            feature_map.append(x)

        output = x

        return output, feature_map


class PeriodDiscriminator(nn.Module):
    """Period discriminator.

    Args:
        period (int): Period of reshaped waveform.
        num_features (list or int): Number of features in convolution.
        kernel_size (list or _size_1_t): Kernel size in convolution.
        stride (list or _size_1_t): Stride in convolution.
        groups (list or int): Number of groups in convolution.
        negative_slope (float): Negative slope in leaky relu.
        weight_regularization (list, optional): Weight regularization method.
            Only ``weight_norm`` and ``spectral_norm`` are supported.

    """

    def __init__(
        self,
        period: int,
        num_features: List[int],
        kernel_size: Union[List[_size_1_t], _size_1_t],
        stride: Union[List[_size_1_t], _size_1_t] = 1,
        groups: Union[List[int], int] = 1,
        negative_slope: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        num_layers = len(num_features)

        if type(kernel_size) is int:
            kernel_size = [kernel_size] * num_layers
        else:
            assert len(kernel_size) == num_layers

        if type(stride) is int:
            stride = [stride] * num_layers
        else:
            assert len(stride) == num_layers

        if type(groups) is int:
            groups = [groups] * num_layers
        else:
            assert len(groups) == num_layers

        if weight_regularization is not None:
            assert weight_regularization in available_weight_regularizations

        net = []

        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                # Last layer does not use leaky relu.
                out_channels = 1
                _negative_slope = 1
            else:
                out_channels = num_features[layer_idx + 1]
                _negative_slope = negative_slope

            in_channels = num_features[layer_idx]
            _kernel_size = kernel_size[layer_idx]
            _stride = stride[layer_idx]
            _groups = groups[layer_idx]

            net.append(
                ConvBlock2d(
                    in_channels,
                    out_channels,
                    kernel_size=(_kernel_size, 1),
                    stride=(_stride, 1),
                    groups=_groups,
                    negative_slope=_negative_slope,
                    weight_regularization=weight_regularization,
                    nonlinear_first=False,
                )
            )

        self.net = nn.ModuleList(net)

        self.period = period
        self.num_layers = num_layers

    def weight_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].weight_norm_()

    def remove_weight_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].remove_weight_norm_()

    def spectral_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].spectral_norm_()

    def remove_spectral_norm_(self) -> None:
        for layer_idx in range(self.num_layers):
            self.net[layer_idx].remove_spectral_norm_()

    @classmethod
    def build_from_default_config(
        cls, period: int, weight_regularization: Optional[str] = "weight_norm"
    ) -> "PeriodDiscriminator":
        num_features = [1, 32, 128, 512, 1024, 1024]
        kernel_size = [5, 5, 5, 5, 5, 3]
        stride = [3, 3, 3, 3, 1, 1]
        groups = [1, 1, 1, 1, 1, 1]
        negative_slope = 0.1

        discriminator = cls(
            period,
            num_features,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            negative_slope=negative_slope,
            weight_regularization=weight_regularization,
        )

        return discriminator

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass of scale discriminator.

        Args:
            input (torch.Tensor): Waveform-like feature of shape (batch_size, in_channels, length),
                where in_channels should be equal to ``num_features[0]``.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Downsampled feature of shape (batch_size, 1, length // down_scale).
                - list: List of torch.Tensor. Nth item is output of nth layer.

        """
        period = self.period
        batch_size, in_channels, length = input.size()

        x = input
        feature_map = []

        padding = (period - length % period) % period
        x = F.pad(input, (0, padding), mode="reflect")
        x = x.view(batch_size, in_channels, -1, period)

        for layer_idx in range(self.num_layers):
            x = self.net[layer_idx](x)
            feature_map.append(x)

        # number of output channels is fixed as 1.
        output = x.view(batch_size, 1, -1)

        return output, feature_map


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


class HiFiGANGenerator(Generator):
    """Alias of Generator for HiFi-GAN."""


class HiFiGANDiscriminator(Discriminator):
    """Alias of Discriminator for HiFi-GAN."""
