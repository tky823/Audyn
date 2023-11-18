from typing import Optional

import pytest
import torch
import torch.nn as nn
from packaging import version
from torch.nn.common_types import _size_2_t

from audyn.modules.pixelsnail import CausalConv2d, PointwiseConvBlock2d, ResidualBlock2d

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")

parameters_kernel_size = [2, (3, 2)]
parameters_weight_regularization = [
    None,
    "weight_norm",
    "spectral_norm",
]
parameters_capture_center = [True, False]


@pytest.mark.parametrize("kernel_size", parameters_kernel_size)
@pytest.mark.parametrize("weight_regularization", parameters_weight_regularization)
def test_residual_block2d(kernel_size: _size_2_t, weight_regularization: Optional[str]) -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features = 4
    height, width = 5, 7

    module = ResidualBlock2d(
        num_features,
        kernel_size=kernel_size,
        weight_regularization=weight_regularization,
    )
    input = torch.randn((batch_size, num_features, height, width))
    output = module(input)

    assert output.size() == input.size()

    if weight_regularization is not None:
        if weight_regularization == "weight_norm":
            module.remove_weight_norm_()
        elif weight_regularization == "spectral_norm":
            module.remove_spectral_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )

    output = module(input)

    assert output.size() == input.size()


@pytest.mark.parametrize("kernel_size", parameters_kernel_size)
def test_causal_conv2d(kernel_size: _size_2_t) -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 4, 3
    height, width = 5, 7

    module = CausalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
    )
    input = torch.randn((batch_size, in_channels, height, width))
    output = module(input)

    # weight normalization
    if IS_TORCH_LT_2_1:
        weight_norm_fn = nn.utils.weight_norm
    else:
        weight_norm_fn = nn.utils.parametrizations.weight_norm

    module_weight_norm = weight_norm_fn(module)
    output_weight_norm = module_weight_norm(input)

    assert torch.allclose(output, output_weight_norm)

    # spectral normalization
    if IS_TORCH_LT_2_1:
        spectral_norm_fn = nn.utils.spectral_norm
    else:
        spectral_norm_fn = nn.utils.parametrizations.spectral_norm

    module_spectral_norm = spectral_norm_fn(module)
    _ = module_spectral_norm(input)


@pytest.mark.parametrize("weight_regularization", parameters_weight_regularization)
def test_pointwise_convblock2d(weight_regularization: Optional[str]) -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features = 4
    height, width = 5, 7

    module = PointwiseConvBlock2d(
        num_features,
        weight_regularization=weight_regularization,
    )
    input = torch.randn((batch_size, num_features, height, width))
    output = module(input)

    assert output.size() == input.size()

    if weight_regularization is not None:
        if weight_regularization == "weight_norm":
            module.remove_weight_norm_()
        elif weight_regularization == "spectral_norm":
            module.remove_spectral_norm_()
        else:
            raise ValueError(
                "{}-based weight regularization is not supported.".format(weight_regularization)
            )
