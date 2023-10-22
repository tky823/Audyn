import torch
import torch.nn as nn
from dummy import allclose

from audyn.modules.waveglow import StackedResidualConvBlock1d, WaveNetAffineCoupling

batch_size = 2
length = 64


def test_waveglow_affine_coupling():
    torch.manual_seed(0)

    coupling_channels, hidden_channels = 4, 6
    local_dim = 2
    num_layers = 3

    input = torch.randn((batch_size, 2 * coupling_channels, length))
    local_conditioning = torch.randn((batch_size, local_dim, length))

    model = WaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
        local_dim=local_dim,
    )

    nn.init.normal_(model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.coupling.bottleneck_conv1d_out.bias.data)

    z = model(input, local_conditioning=local_conditioning)
    output = model(z, local_conditioning=local_conditioning, reverse=True)

    assert output.size() == input.size()
    allclose(output, input, atol=1e-6)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        local_conditioning=local_conditioning,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        local_conditioning=local_conditioning,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-6)
    allclose(logdet, zeros)


def test_stacked_residual_conv_block():
    torch.manual_seed(0)

    in_channels, hidden_channels = 4, 6
    local_dim = 2
    num_layers = 3

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn((batch_size, local_dim, length))

    model = StackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        local_dim,
        num_layers=num_layers,
        local_dim=local_dim,
    )

    log_s, t = model(input, local_conditioning=local_conditioning)

    assert log_s.size() == (batch_size, in_channels, length)
    assert t.size() == (batch_size, in_channels, length)
