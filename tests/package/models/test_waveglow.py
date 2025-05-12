import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.models.waveglow import (
    MultiSpeakerWaveGlow,
    StackedWaveGlowBlock,
    WaveGlow,
    WaveGlowBlock,
)

batch_size = 2
length = 64


def test_waveglow():
    torch.manual_seed(0)

    in_channels = 6
    n_mels = 5
    hop_length = 4

    hidden_channels = 4
    num_flows, num_stacks, num_layers = 2, 2, 3

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn(batch_size, n_mels, length // hop_length)

    upsample = nn.ConvTranspose1d(
        n_mels,
        n_mels,
        kernel_size=length // hop_length,
        stride=length // hop_length,
    )
    model = WaveGlow(
        in_channels,
        hidden_channels,
        num_flows=num_flows,
        num_stacks=num_stacks,
        num_layers=num_layers,
        upsample=upsample,
        local_channels=n_mels,
    )

    z = model(input, local_conditioning=local_conditioning)
    output = model(z, local_conditioning=local_conditioning, reverse=True)

    assert output.size() == input.size()
    allclose(output, input, atol=1e-5)

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
    allclose(output, input, atol=1e-5)
    allclose(logdet, zeros)


def test_multispk_waveglow():
    torch.manual_seed(0)

    in_channels = 6
    n_mels = 5
    hop_length = 4

    hidden_channels = 4
    num_flows, num_stacks, num_layers = 2, 2, 3

    num_speakers = 4
    padding_idx = 0
    embed_dim = 5

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn(batch_size, n_mels, length // hop_length)
    speaker = torch.randint(0, num_speakers, (batch_size,), dtype=torch.long)

    upsample = nn.ConvTranspose1d(
        n_mels,
        n_mels,
        kernel_size=length // hop_length,
        stride=length // hop_length,
    )
    speaker_encoder = nn.Embedding(
        num_embeddings=num_speakers,
        embedding_dim=embed_dim,
        padding_idx=padding_idx,
    )
    model = MultiSpeakerWaveGlow(
        in_channels,
        hidden_channels,
        num_flows=num_flows,
        num_stacks=num_stacks,
        num_layers=num_layers,
        upsample=upsample,
        speaker_encoder=speaker_encoder,
        local_channels=n_mels,
        global_channels=embed_dim,
    )

    z = model(
        input,
        local_conditioning=local_conditioning,
        speaker=speaker,
    )
    output = model(
        z,
        local_conditioning=local_conditioning,
        speaker=speaker,
        reverse=True,
    )

    assert output.size() == input.size()
    allclose(output, input, atol=1e-5)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        local_conditioning=local_conditioning,
        speaker=speaker,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        local_conditioning=local_conditioning,
        speaker=speaker,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-5)
    allclose(logdet, zeros)


def test_stacked_waveglow_block():
    torch.manual_seed(0)

    in_channels, hidden_channels = 8, 6
    local_channels = 2
    num_stacks, num_layers = 2, 3

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn((batch_size, local_channels, length))

    model = StackedWaveGlowBlock(
        in_channels,
        hidden_channels,
        num_stacks=num_stacks,
        num_layers=num_layers,
        local_channels=local_channels,
    )

    for m in model.backbone:
        nn.init.normal_(m.affine_coupling.coupling.bottleneck_conv1d_out.weight.data)
        nn.init.normal_(m.affine_coupling.coupling.bottleneck_conv1d_out.bias.data)

    z = model(input, local_conditioning=local_conditioning)
    output = model(z, local_conditioning=local_conditioning, reverse=True)

    assert output.size() == input.size()
    allclose(output, input, atol=1e-5)

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
    allclose(output, input, atol=1e-5)
    allclose(logdet, zeros, atol=1e-4)


def test_waveglow_block():
    torch.manual_seed(0)

    in_channels, hidden_channels = 8, 6
    local_channels = 2
    num_layers = 3

    input = torch.randn((batch_size, in_channels, length))
    local_conditioning = torch.randn((batch_size, local_channels, length))

    model = WaveGlowBlock(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        local_channels=local_channels,
    )

    nn.init.normal_(model.affine_coupling.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.affine_coupling.coupling.bottleneck_conv1d_out.bias.data)

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
    allclose(logdet, zeros, atol=1e-4)
