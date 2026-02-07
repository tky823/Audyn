import torch

from audyn.models.wavenext import WaveNeXtVocoder


def test_wavenext_vocoder() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels, out_channels, hidden_channels, bottleneck_channels = 80, 1, 4, 8
    kernel_size = 7
    post_kernel_size = 64
    num_blocks = 2
    length = 32

    model = WaveNeXtVocoder(
        in_channels,
        out_channels,
        hidden_channels,
        bottleneck_channels,
        kernel_size=kernel_size,
        post_kernel_size=post_kernel_size,
        num_blocks=num_blocks,
    )
    input = torch.randn((batch_size, in_channels, length))
    output = model(input)

    upsampled_length = model.up_scale * length

    assert output.size() == (batch_size, out_channels, upsampled_length)

    post_stride = 16

    model = WaveNeXtVocoder(
        in_channels,
        out_channels,
        hidden_channels,
        bottleneck_channels,
        kernel_size=kernel_size,
        post_kernel_size=post_kernel_size,
        post_stride=post_stride,
        num_blocks=num_blocks,
    )
    output = model(input)

    upsampled_length = model.up_scale * length

    assert output.size() == (batch_size, out_channels, upsampled_length)
