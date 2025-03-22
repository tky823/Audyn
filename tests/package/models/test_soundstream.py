import pytest
import torch
from dummy import allclose

from audyn.models.soundstream import (
    Decoder,
    Encoder,
    SoundStream,
    SpectrogramDiscriminator,
)


@pytest.mark.parametrize("is_causal", [True, False])
def test_soundstream(is_causal: bool) -> None:
    torch.manual_seed(0)

    in_channels, embedding_dim, hidden_channels = 1, 5, 2
    depth_rate = 2
    kernel_size_out = kernel_size_in = 3
    kernel_size = 3
    stride, dilation_rate = [2, 4, 5], 2
    num_layers = 2
    codebook_size = 16
    num_stages = 4

    batch_size, compressed_length = 3, 8
    input_length = compressed_length

    for s in stride:
        input_length *= s

    encoder = Encoder(
        in_channels,
        embedding_dim,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        stride=stride,
        dilation_rate=dilation_rate,
        num_layers=num_layers,
        is_causal=is_causal,
    )
    decoder = Decoder(
        embedding_dim,
        in_channels,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_out,
        kernel_size_out=kernel_size_in,
        kernel_size=kernel_size,
        stride=stride[-1::-1],
        dilation_rate=dilation_rate,
        num_layers=num_layers,
        is_causal=is_causal,
    )
    model = SoundStream(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_stages=num_stages,
        dropout=False,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output, encoded, hierarchical_quantized, hierarchical_residual, indices = model(input)

    assert output.size() == input.size()
    assert encoded.size() == (batch_size, embedding_dim, compressed_length)
    assert hierarchical_quantized.size() == (
        batch_size,
        num_stages,
        embedding_dim,
        compressed_length,
    )
    assert hierarchical_residual.size() == hierarchical_quantized.size()
    assert indices.size() == (batch_size, num_stages, compressed_length)

    hierarchical_quantized = hierarchical_quantized.transpose(1, 0)
    hierarchical_residual = hierarchical_residual.transpose(1, 0)

    for stage_idx in range(num_stages):
        _hierarchical_residual = hierarchical_residual[stage_idx]

        if stage_idx == 0:
            allclose(_hierarchical_residual, encoded)
        else:
            _hierarchical_quantized = torch.sum(hierarchical_quantized[:stage_idx], dim=0)

            allclose(_hierarchical_quantized + _hierarchical_residual, encoded, atol=1e-7)

    # initialization
    model = SoundStream(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_stages=num_stages,
        dropout=False,
        init_by_kmeans=10,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output, encoded, hierarchical_quantized, hierarchical_residual, indices = model(input)

    assert output.size() == input.size()
    assert encoded.size() == (batch_size, embedding_dim, compressed_length)
    assert hierarchical_quantized.size() == (
        batch_size,
        num_stages,
        embedding_dim,
        compressed_length,
    )
    assert hierarchical_residual.size() == hierarchical_quantized.size()
    assert indices.size() == (batch_size, num_stages, compressed_length)

    hierarchical_quantized = hierarchical_quantized.transpose(1, 0)
    hierarchical_residual = hierarchical_residual.transpose(1, 0)

    for stage_idx in range(num_stages):
        _hierarchical_residual = hierarchical_residual[stage_idx]

        if stage_idx == 0:
            allclose(_hierarchical_residual, encoded)
        else:
            _hierarchical_quantized = torch.sum(hierarchical_quantized[:stage_idx], dim=0)

            allclose(_hierarchical_quantized + _hierarchical_residual, encoded, atol=1e-7)


@pytest.mark.parametrize("is_causal", [True, False])
def test_soundstream_encoder(is_causal: bool) -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 1, 5, 2
    depth_rate = 2
    kernel_size_out = kernel_size_in = 3
    kernel_size = 3
    stride, dilation_rate = [2, 4, 5], 2
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
        is_causal=is_causal,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output = encoder(input)

    assert output.size() == (batch_size, out_channels, output_length)


@pytest.mark.parametrize("is_causal", [True, False])
def test_soundstream_decoder(is_causal: bool) -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 5, 1, 2
    depth_rate = 2
    kernel_size_out = kernel_size_in = 3
    kernel_size = 3
    stride, dilation_rate = [5, 4, 2], 2
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
        is_causal=is_causal,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output = decoder(input)

    assert output.size() == (batch_size, out_channels, output_length)


def test_soundstream_spectrogram_discriminator() -> None:
    torch.manual_seed(0)

    num_features = [4, 4, 8, 8]
    kernel_size_in, kernel_size = (3, 5), 3
    down_scale = [2, 4, 8]
    n_fft, hop_length = 1024, 256
    kernel_size_out = n_fft // 2

    for s in down_scale:
        kernel_size_out //= s

    kernel_size_out = (kernel_size_out, 1)

    discriminator = SpectrogramDiscriminator(
        num_features,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        down_scale=down_scale,
        transform=True,
        transform_kwargs={"n_fft": n_fft, "hop_length": hop_length, "return_complex": True},
    )

    batch_size, in_channels, length = 4, 1, 40000

    input = torch.randn((batch_size, in_channels, length))
    output, feature_map = discriminator(input)

    assert output.size(0) == batch_size
    assert output.size(1) == 1

    for _feature_map in feature_map:
        assert _feature_map.size(0) == batch_size
