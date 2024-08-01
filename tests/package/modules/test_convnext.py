import pytest
import torch
import torch.nn as nn

from audyn.modules.convnext import ConvNeXtBlock1d, StackedConvNeXtBlock1d


def test_stacked_convnext_block() -> None:
    torch.manual_seed(0)

    num_features, bottleneck_channels = 3, 5
    kernel_size = 7
    norm = nn.LayerNorm(num_features)
    activation = "gelu"
    num_blocks = 3
    batch_size, length = 4, 32

    module = StackedConvNeXtBlock1d(
        num_features,
        bottleneck_channels,
        kernel_size,
        norm,
        activation=activation,
        num_blocks=num_blocks,
    )
    input = torch.randn((batch_size, num_features, length))
    output = module(input)

    assert output.size() == input.size()


@pytest.mark.parametrize("is_str_norm", [True, False])
def test_convnext_block(is_str_norm: bool) -> None:
    torch.manual_seed(0)

    num_features, bottleneck_channels = 3, 5
    kernel_size = 7

    if is_str_norm:
        norm = "layer_norm"
    else:
        norm = nn.LayerNorm(num_features)

    activation = "gelu"
    scale = 1 / 6
    batch_size, length = 4, 32

    module = ConvNeXtBlock1d(
        num_features,
        bottleneck_channels,
        kernel_size,
        norm,
        activation=activation,
        scale=scale,
    )
    input = torch.randn((batch_size, num_features, length))
    output = module(input)

    assert output.size() == input.size()
