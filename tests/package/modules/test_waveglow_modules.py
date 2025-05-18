import pytest
import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.modules.waveglow import StackedResidualConvBlock1d, WaveNetAffineCoupling

batch_size = 2
length = 64


@pytest.mark.parametrize("scaling", [True, False])
@pytest.mark.parametrize("set_scaling_channels", [True, False])
def test_waveglow_affine_coupling(scaling: bool, set_scaling_channels: bool) -> None:
    torch.manual_seed(0)

    coupling_channels, hidden_channels = 4, 6
    local_channels = 2
    num_layers = 3

    if set_scaling_channels:
        if not scaling:
            pytest.skip("Pair of scaling=False and set_scaling_channels=True is not supported.")

        scaling_channels = coupling_channels
    else:
        scaling_channels = None

    input = torch.randn((batch_size, 2 * coupling_channels, length))
    local_conditioning = torch.randn((batch_size, local_channels, length))

    model = WaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
        local_channels=local_channels,
        scaling=scaling,
        scaling_channels=scaling_channels,
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
    local_channels = 2
    num_layers = 3

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn((batch_size, local_channels, length))

    model = StackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        local_channels,
        num_layers=num_layers,
        local_channels=local_channels,
    )

    log_s, t = model(input, local_conditioning=local_conditioning)

    assert log_s.size() == (batch_size, in_channels, length)
    assert t.size() == (batch_size, in_channels, length)
