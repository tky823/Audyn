import torch

from audyn.models.soundstream import Decoder, Encoder


def test_soundstream_encoder() -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 1, 5, 2
    depth_rate = 2
    kernel_size_out = kernel_size_in = 3
    kernel_size = 3
    stride = [2, 4, 5]
    dilation_rate = 2
    num_layers = 2

    batch_size, output_length = 3, 8
    input_length = output_length

    for s in stride:
        input_length *= s

    encoder = Encoder(
        in_channels,
        out_channels,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        stride=stride,
        dilation_rate=dilation_rate,
        num_layers=num_layers,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output = encoder(input)

    assert output.size() == (batch_size, out_channels, output_length)


def test_soundstream_decoder() -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 5, 1, 2
    depth_rate = 2
    kernel_size_out = kernel_size_in = 3
    kernel_size = 3
    stride = [5, 4, 2]
    dilation_rate = 2
    num_layers = 2

    batch_size, input_length = 3, 8
    output_length = input_length

    for s in stride:
        output_length *= s

    decoder = Decoder(
        in_channels,
        out_channels,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        stride=stride,
        dilation_rate=dilation_rate,
        num_layers=num_layers,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output = decoder(input)

    assert output.size() == (batch_size, out_channels, output_length)
