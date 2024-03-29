import pytest
import torch

from audyn.models.encodec import Decoder, EnCodec, Encoder


@pytest.mark.parametrize("is_causal", [True, False])
def test_official_encodec(is_causal: bool) -> None:
    torch.manual_seed(0)

    is_causal = True

    in_channels, embedding_dim, hidden_channels = 1, 128, 32
    depth_rate = 2
    kernel_size_out = kernel_size_in = 7
    kernel_size = 3
    stride = [2, 4, 5, 8]
    num_rnn_layers = 2
    codebook_size = 1024
    num_stages = 32

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
        num_rnn_layers=num_rnn_layers,
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
        num_rnn_layers=num_rnn_layers,
        is_causal=is_causal,
    )
    model = EnCodec(
        encoder,
        decoder,
        codebook_size=codebook_size,
        embedding_dim=embedding_dim,
        num_stages=num_stages,
        dropout=False,
    )

    input = torch.randn((batch_size, in_channels, input_length))
    output, encoded, hierarchical_quantized, indices = model(input)

    assert output.size() == input.size()
    assert encoded.size() == (batch_size, embedding_dim, compressed_length)
    assert hierarchical_quantized.size() == (
        batch_size,
        num_stages,
        embedding_dim,
        compressed_length,
    )
    assert indices.size() == (batch_size, num_stages, compressed_length)

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    assert num_parameters == 18870114


def test_encodec_encoder() -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 1, 128, 32
    depth_rate = 2
    kernel_size_in, kernel_size_out, kernel_size = 7, 7, 3
    stride = [2, 4, 5, 8]

    batch_size = 4
    length_in = 1600

    input = torch.randn((batch_size, in_channels, length_in))

    encoder = Encoder(
        in_channels,
        out_channels,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        stride=stride,
    )
    output = encoder(input)

    length_out = length_in

    for s in stride:
        length_out //= s

    assert output.size() == (batch_size, out_channels, length_out)


def test_encodec_decoder() -> None:
    in_channels, out_channels, hidden_channels = 128, 1, 32
    depth_rate = 2
    kernel_size_in, kernel_size_out, kernel_size = 7, 7, 3
    stride = [8, 5, 4, 2]

    batch_size = 4
    length_in = 100

    input = torch.randn((batch_size, in_channels, length_in))

    decoder = Decoder(
        in_channels,
        out_channels,
        hidden_channels,
        depth_rate=depth_rate,
        kernel_size_in=kernel_size_in,
        kernel_size_out=kernel_size_out,
        kernel_size=kernel_size,
        stride=stride,
    )
    output = decoder(input)

    length_out = length_in

    for s in stride:
        length_out *= s

    assert output.size() == (batch_size, out_channels, length_out)
