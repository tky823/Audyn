import torch
import torch.nn as nn
from dummy import allclose

from audyn.modules.glow import ActNorm1d, InvertiblePointwiseConv1d
from audyn.modules.glowtts import (
    MaskedActNorm1d,
    MaskedInvertiblePointwiseConv1d,
    MaskedStackedResidualConvBlock1d,
    MaskedWaveNetAffineCoupling,
)


def test_masked_act_norm1d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features = 6
    max_length = 16

    # w/ 2D padding mask
    model = MaskedActNorm1d(num_features)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    non_padding_mask = torch.logical_not(padding_mask)
    num_elements_per_channel = non_padding_mask.sum()

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)

    # w/ 3D padding mask
    batch_size = 4
    num_features = 3
    max_length = 6

    model = MaskedActNorm1d(num_features)

    length = torch.randint(
        num_features,
        num_features * max_length + 1,
        (batch_size,),
        dtype=torch.long,
    )
    max_length = torch.max(length)
    max_length = max_length + (num_features - max_length % num_features) % num_features
    input = torch.randn((batch_size, max_length))
    input = input.view(batch_size, num_features, -1)
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    padding_mask = padding_mask.view(batch_size, num_features, -1)
    non_padding_mask = torch.logical_not(padding_mask)
    num_elements_per_channel = non_padding_mask.sum(dim=(0, 2))

    input = input.masked_fill(padding_mask, 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)

    # w/o padding mask
    batch_size = 2
    num_features = 6
    max_length = 16

    masked_model = MaskedActNorm1d(num_features)
    non_masked_model = ActNorm1d(num_features)

    input = torch.randn(batch_size, num_features, max_length)

    masked_z = masked_model(input)
    masked_output = masked_model(masked_z, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        logdet=zeros,
    )
    masked_output, masked_logdet = masked_model(
        masked_z,
        logdet=masked_z_logdet,
        reverse=True,
    )
    non_masked_z, non_masked_z_logdet = non_masked_model(
        input,
        logdet=zeros,
    )
    non_masked_output, non_masked_logdet = non_masked_model(
        non_masked_z,
        logdet=non_masked_z_logdet,
        reverse=True,
    )
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)


def test_masked_invertible_pointwise_conv1d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features, num_splits = 8, 4
    max_length = 16

    # w/ 2D padding mask
    model = MaskedInvertiblePointwiseConv1d(num_splits)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input, atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-7)
    allclose(logdet, zeros)

    # w/o padding mask
    batch_size = 2
    num_features = num_splits = 6
    max_length = 16

    masked_model = MaskedInvertiblePointwiseConv1d(num_splits)
    non_masked_model = InvertiblePointwiseConv1d(num_features)
    non_masked_model.weight.data.copy_(masked_model.weight.data.detach())

    input = torch.randn(batch_size, num_features, max_length)

    masked_z = masked_model(input)
    masked_output = masked_model(masked_z, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        logdet=zeros,
    )
    masked_output, masked_logdet = masked_model(
        masked_z,
        logdet=masked_z_logdet,
        reverse=True,
    )
    non_masked_z, non_masked_z_logdet = non_masked_model(
        input,
        logdet=zeros,
    )
    non_masked_output, non_masked_logdet = non_masked_model(
        non_masked_z,
        logdet=non_masked_z_logdet,
        reverse=True,
    )

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)


def test_masked_wavenet_affine_coupling() -> None:
    torch.manual_seed(0)

    batch_size, max_length = 2, 16
    coupling_channels, hidden_channels = 4, 6
    num_layers = 3

    # w/ 2D padding mask
    model = MaskedWaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
    )

    nn.init.normal_(model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.coupling.bottleneck_conv1d_out.bias.data)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, 2 * coupling_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)

    # w/o padding mask
    masked_model = MaskedWaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
    )

    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.bias.data)

    input = torch.randn(batch_size, 2 * coupling_channels, max_length)
    z = masked_model(input)
    output = masked_model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = masked_model(
        input,
        logdet=zeros,
    )
    output, logdet = masked_model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)


def test_stacked_residual_conv_block():
    torch.manual_seed(0)

    batch_size, max_length = 2, 16
    in_channels, hidden_channels = 4, 6
    num_layers = 3

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length).item()
    input = torch.randn((batch_size, in_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    model = MaskedStackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
    )
    nn.init.normal_(model.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.bottleneck_conv1d_out.bias.data)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    log_s, t = model(input, padding_mask=padding_mask)

    assert log_s.size() == (batch_size, in_channels, max_length)
    assert t.size() == (batch_size, in_channels, max_length)
