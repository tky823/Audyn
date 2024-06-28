import torch

from audyn.models.conv_tasnet import ConvTasNet, Separator
from audyn.models.tasnet import Decoder, Encoder


def test_conv_tasnet() -> None:
    torch.manual_seed(0)

    batch_size, timesteps = 3, 512
    in_channels = 1
    bottleneck_channels, hidden_channels, skip_channels = 8, 8, 8
    num_sources = 2
    num_basis = 8

    encoder = Encoder(in_channels, num_basis)
    decoder = Decoder(in_channels, num_basis)
    separator = Separator(
        num_basis,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        num_sources=num_sources,
    )
    model = ConvTasNet(encoder, decoder, separator, num_sources=num_sources)

    input = torch.randn(batch_size, timesteps)
    output = model(input)

    assert output.size() == (batch_size, num_sources, timesteps)

    in_channels = 2
    num_sources = 3

    encoder = Encoder(in_channels, num_basis)
    decoder = Decoder(in_channels, num_basis)
    separator = Separator(
        num_basis,
        bottleneck_channels=bottleneck_channels,
        hidden_channels=hidden_channels,
        skip_channels=skip_channels,
        num_sources=num_sources,
    )
    model = ConvTasNet(encoder, decoder, separator, num_sources=num_sources)

    input = torch.randn(batch_size, in_channels, timesteps)
    output = model(input)

    assert output.size() == (batch_size, num_sources, in_channels, timesteps)
