import pytest
import torch

from audyn.models.hifigan import (
    Generator,
    MultiPeriodDiscriminator,
    MultiReceptiveFieldFusion,
    MultiScaleDiscriminator,
    PeriodDiscriminator,
    ResidualBlock1d,
    ScaleDiscriminator,
    StackedConvBlock1d,
)


@pytest.mark.parametrize("variation", ["v1", "v2", "v3"])
def test_hifigan_generator(variation: str) -> None:
    batch_size = 4
    in_channels, out_channels = 80, 1
    length = 32

    model = Generator.build_from_default_config(variation=variation)
    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    # NOTE: For Python>=3.8, math.prod is usedful.
    upsampled_length = length

    for up_scale in model.up_stride:
        upsampled_length *= up_scale

    assert output.size() == (batch_size, out_channels, upsampled_length)

    if len(model.registered_weight_norms) > 0:
        model.remove_weight_norm_()


def test_multi_receptive_field_fusion() -> None:
    batch_size = 4
    in_channels, out_channels = 512, 256
    kernel_size = [3, 5, 7]
    dilation = [1, 3]
    up_kernel_size, up_stride = 16, 8
    num_blocks, num_layers = 3, 2
    length = 32

    model = MultiReceptiveFieldFusion(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        up_kernel_size=up_kernel_size,
        up_stride=up_stride,
        num_layers=num_layers,
        num_blocks=num_blocks,
    )
    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, length * up_stride)


def test_multi_scale_discriminator() -> None:
    batch_size = 4
    in_channels = 1
    length = 1024

    model = MultiScaleDiscriminator.build_from_default_config()
    input = torch.randn((batch_size, in_channels, length))
    output, feature_map = model(input)

    for discriminator_idx in range(model.num_discriminators):
        assert output[discriminator_idx].size(0) == batch_size

        for layer_idx in range(model.discriminator[discriminator_idx].num_layers):
            _feature_map = feature_map[discriminator_idx][layer_idx]

            assert _feature_map.size(0) == batch_size

    if model.weight_regularization is not None:
        model.remove_weight_regularization_()


def test_multi_period_discriminator() -> None:
    batch_size = 4
    in_channels = 1
    length = 1024

    model = MultiPeriodDiscriminator.build_from_default_config()
    input = torch.randn((batch_size, in_channels, length))
    output, feature_map = model(input)

    for discriminator_idx in range(model.num_discriminators):
        assert output[discriminator_idx].size(0) == batch_size

        for layer_idx in range(model.discriminator[discriminator_idx].num_layers):
            _feature_map = feature_map[discriminator_idx][layer_idx]

            assert _feature_map.size(0) == batch_size

    if model.weight_regularization is not None:
        model.remove_weight_regularization_()


def test_scale_discriminator() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    length = 1024

    model = ScaleDiscriminator.build_from_default_config()
    input = torch.randn((batch_size, in_channels, length))

    with torch.no_grad():
        output, feature_map = model(input)

    assert output.size()[:2] == (batch_size, 1)
    assert len(feature_map) == len(model.net)

    for idx in range(len(feature_map)):
        assert feature_map[idx].size(0) == batch_size


def test_period_discriminator() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    length = 1024

    period = 2

    model = PeriodDiscriminator.build_from_default_config(period)
    input = torch.randn((batch_size, in_channels, length))

    with torch.no_grad():
        output, feature_map = model(input)

    assert output.size()[:2] == (batch_size, 1)
    assert len(feature_map) == len(model.net)

    for idx in range(len(feature_map)):
        assert feature_map[idx].size(0) == batch_size
        assert feature_map[idx].size(-1) == period


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
