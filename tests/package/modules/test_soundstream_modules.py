import torch

from audyn.modules.soundstream import DecoderBlock, EncoderBlock, ResidualUnit1d


def test_soundstream_encoder_block() -> None:
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
    )

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, length // stride)


def test_soundstream_decoder_block() -> None:
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
    )

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == (batch_size, out_channels, stride * length)


def test_soundstream_residual_unit() -> None:
    batch_size = 2
    length = 20
    in_channels = 3
    kernel_size, dilation = 5, 9

    model = ResidualUnit1d(in_channels, kernel_size=kernel_size, dilation=dilation)

    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    assert output.size() == input.size()
