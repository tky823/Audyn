import os

import pytest
import torch
from audyn_test import allclose
from audyn_test.utils import audyn_test_cache_dir

from audyn.models.encodec import (
    Decoder,
    EnCodec,
    Encoder,
    encodec_24khz_codebook_size,
    encodec_24khz_num_codebooks,
    encodec_32khz_codebook_size,
    encodec_32khz_num_codebooks,
)
from audyn.utils._github import download_file_from_github_release


@pytest.mark.parametrize("is_causal", [True, False])
def test_official_encodec(is_causal: bool) -> None:
    torch.manual_seed(0)

    in_channels = 1
    embedding_dim, hidden_channels = 128, 32
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

            allclose(_hierarchical_quantized + _hierarchical_residual, encoded, atol=1e-5)

    if is_causal:
        # regression test
        for sample_rate in [24, 32]:
            url = f"https://github.com/tky823/Audyn/releases/download/v0.2.1/test_official_encodec_{sample_rate}kHz.pth"  # noqa: E501

            filename = os.path.basename(url)
            path = os.path.join(audyn_test_cache_dir, filename)
            download_file_from_github_release(url, path)

            data = torch.load(path, weights_only=True)

            model = EnCodec.build_from_pretrained(f"encodec_{sample_rate}khz")

            if sample_rate == 24:
                assert model.num_codebooks == encodec_24khz_num_codebooks
                assert model.codebook_size == encodec_24khz_codebook_size
            elif sample_rate == 32:
                assert model.num_codebooks == encodec_32khz_num_codebooks
                assert model.codebook_size == encodec_32khz_codebook_size
            else:
                raise ValueError(f"Unsupported sample rate {sample_rate}.")

            input = data["input"]
            expected_output = data["output"]

            model.eval()

            with torch.no_grad():
                output, encoded, hierarchical_quantized, hierarchical_residual, indices = model(
                    input
                )

            allclose(output, expected_output, atol=1e-5)

            num_parameters = 0

            for p in model.parameters():
                if p.requires_grad:
                    num_parameters += p.numel()

            if sample_rate == 24:
                assert num_parameters == 19046114
            elif sample_rate == 32:
                assert num_parameters == 57933250
            else:
                raise ValueError(f"Unsupported sample rate {sample_rate}.")


def test_encodec_encoder() -> None:
    torch.manual_seed(0)

    in_channels, out_channels, hidden_channels = 1, 32, 8
    depth_rate = 2
    kernel_size_in, kernel_size_out, kernel_size = 5, 5, 3
    stride = [2, 5]

    batch_size = 4
    length_in = 400

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
    in_channels, out_channels, hidden_channels = 32, 1, 8
    depth_rate = 2
    kernel_size_in, kernel_size_out, kernel_size = 5, 5, 3
    stride = [5, 2]

    batch_size = 4
    length_in = 40

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
