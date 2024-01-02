import pytest
import torch

from audyn.modules.soundstream import DecoderBlock, EncoderBlock, ResidualUnit1d, ResidualUnit2d


@pytest.mark.parametrize("causal", [True, False])
def test_soundstream_encoder_block(causal: bool) -> None:
    batch_size, length = 2, 20
    in_channels, out_channels = 3, 6
    kernel_size, stride, dilation_rate = 5, 2, 2
    num_layers = 3

    model = EncoderBlock(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation_rate=dilation_rate,
        num_layers=num_layers,
        causal=causal,
    )

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, length // stride)


@pytest.mark.parametrize("causal", [True, False])
def test_soundstream_decoder_block(causal: bool) -> None:
    batch_size, length = 2, 20
    in_channels, out_channels = 6, 3
    kernel_size, stride, dilation_rate = 5, 2, 2
    num_layers = 3

    model = DecoderBlock(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation_rate=dilation_rate,
        num_layers=num_layers,
        causal=causal,
    )

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, stride * length)


@pytest.mark.parametrize("causal", [True, False])
def test_soundstream_residual_unit1d(causal: bool) -> None:
    batch_size = 2
    length = 20
    in_channels = 3
    kernel_size, dilation = 5, 9

    model = ResidualUnit1d(
        in_channels,
        kernel_size=kernel_size,
        dilation=dilation,
        causal=causal,
    )

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == input.size()


def test_soundstream_residual_unit2d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    height, width = 20, 30
    in_channels, out_channels = 3, 5
    kernel_size, down_scale = (3, 3), (2, 1)

    model = ResidualUnit2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        down_scale=down_scale,
    )

    input = torch.randn((batch_size, in_channels, height, width))
    output = model(input)

    assert output.size() == (
        batch_size,
        out_channels,
        height // down_scale[0],
        width // down_scale[1],
    )
