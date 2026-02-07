import torch

from audyn.modules.encodec import DecoderBlock, EncoderBlock


def test_encodec_encoder_block() -> None:
    torch.manual_seed(0)

    in_channels = 4
    out_channels = 2 * in_channels
    kernel_size = 3
    stride = 2

    batch_size = 4
    length = stride * 100

    input = torch.randn((batch_size, in_channels, length))

    model = EncoderBlock(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        is_causal=True,
    )
    output = model(input)

    assert output.size() == (batch_size, out_channels, length // stride)


def test_encodec_decoder_block() -> None:
    torch.manual_seed(0)

    out_channels = 32
    in_channels = 2 * out_channels
    kernel_size = 3
    stride = 2

    batch_size = 4
    length = 100

    input = torch.randn((batch_size, in_channels, length))

    model = DecoderBlock(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        is_causal=True,
    )
    output = model(input)

    assert output.size() == (batch_size, out_channels, stride * length)
