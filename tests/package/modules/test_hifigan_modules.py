import pytest
import torch

from audyn.modules.hifigan import ResidualBlock1d, StackedConvBlock1d


def test_residual_block1d() -> None:
    batch_size = 4
    num_features = 3
    kernel_size, dilation = [3, 5], [1, 3]
    stacked = True
    length = 1024

    model = ResidualBlock1d(
        num_features,
        kernel_size=kernel_size,
        dilation=dilation,
        stacked=stacked,
        num_layers=len(kernel_size),
    )
    input = torch.randn((batch_size, num_features, length))
    output = model(input)

    assert output.size() == (batch_size, num_features, length)


@pytest.mark.parametrize("weight_regularization", ["weight_norm", "spectral_norm", None])
def test_stacked_conv_block1d(weight_regularization: str) -> None:
    batch_size = 4
    in_channels, out_channels, hidden_channels = 2, 3, 5
    kernel_size, dilation = 3, 1
    nonlinear_first = True
    length = 1024

    model = StackedConvBlock1d(
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        weight_regularization=weight_regularization,
        nonlinear_first=nonlinear_first,
    )
    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, length)

    if weight_regularization == "weight_norm":
        model.remove_weight_norm_()
    elif weight_regularization == "spectral_norm":
        model.remove_spectral_norm_()
