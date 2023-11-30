import pytest
import torch
import torch.nn as nn
from packaging import version
from torch.nn.common_types import _size_2_t

from audyn.modules.pixelcnn import CausalConv2d, HorizontalConv2d, VerticalConv2d

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")

parameters_kernel_size = [3, (5, 3)]
parameters_capture_center = [True, False]


@pytest.mark.parametrize("kernel_size", parameters_kernel_size)
@pytest.mark.parametrize("capture_center", parameters_capture_center)
def test_causal_conv2d(kernel_size: _size_2_t, capture_center: bool) -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 2, 3
    height, width = 5, 7

    input = torch.randn((batch_size, in_channels, height, width))

    # weight normalization
    module = CausalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )
    output = module(input)

    if IS_TORCH_LT_2_1:
        weight_norm_fn = nn.utils.weight_norm
    else:
        weight_norm_fn = nn.utils.parametrizations.weight_norm

    module_weight_norm = weight_norm_fn(module)
    output_weight_norm = module_weight_norm(input)

    assert torch.allclose(output, output_weight_norm)

    # spectral normalization
    module = CausalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )

    if IS_TORCH_LT_2_1:
        spectral_norm_fn = nn.utils.spectral_norm
    else:
        spectral_norm_fn = nn.utils.parametrizations.spectral_norm

    module_spectral_norm = spectral_norm_fn(module)
    _ = module_spectral_norm(input)


@pytest.mark.parametrize("kernel_size", parameters_kernel_size)
@pytest.mark.parametrize("capture_center", parameters_capture_center)
def test_vertical_conv2d(kernel_size: _size_2_t, capture_center: bool) -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 2, 3
    height, width = 5, 7

    input = torch.randn((batch_size, in_channels, height, width))

    # weight normalization
    module = VerticalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )
    output = module(input)

    if IS_TORCH_LT_2_1:
        weight_norm_fn = nn.utils.weight_norm
    else:
        weight_norm_fn = nn.utils.parametrizations.weight_norm

    module_weight_norm = weight_norm_fn(module)
    output_weight_norm = module_weight_norm(input)

    assert torch.allclose(output, output_weight_norm)

    # spectral normalization
    module = VerticalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )

    if IS_TORCH_LT_2_1:
        spectral_norm_fn = nn.utils.spectral_norm
    else:
        spectral_norm_fn = nn.utils.parametrizations.spectral_norm

    module_spectral_norm = spectral_norm_fn(module)
    _ = module_spectral_norm(input)


@pytest.mark.parametrize("capture_center", parameters_capture_center)
def test_horizontal_conv2d(capture_center: bool) -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, out_channels = 2, 3
    kernel_size = 5
    height, width = 5, 7

    input = torch.randn((batch_size, in_channels, height, width))

    # weight normalization
    module = HorizontalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )
    output = module(input)

    if IS_TORCH_LT_2_1:
        weight_norm_fn = nn.utils.weight_norm
    else:
        weight_norm_fn = nn.utils.parametrizations.weight_norm

    module_weight_norm = weight_norm_fn(module)
    output_weight_norm = module_weight_norm(input)

    assert torch.allclose(output, output_weight_norm)

    # spectral normalization
    module = HorizontalConv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        capture_center=capture_center,
    )

    if IS_TORCH_LT_2_1:
        spectral_norm_fn = nn.utils.spectral_norm
    else:
        spectral_norm_fn = nn.utils.parametrizations.spectral_norm

    module_spectral_norm = spectral_norm_fn(module)
    _ = module_spectral_norm(input)
