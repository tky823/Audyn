from typing import List

import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.models.fastspeech import LengthRegulator
from audyn.models.glowtts import (
    Decoder,
    GlowBlock,
    GlowTTS,
    GlowTTSTransformerEncoder,
    TextEncoder,
)
from audyn.modules.duration_predictor import FastSpeechDurationPredictor
from audyn.modules.fastspeech import FFTrBlock

parameters_batch_first = [True, False]


def test_glowtts() -> None:
    torch.manual_seed(0)

    batch_first = True
    batch_size = 4
    num_embeddings = 5
    padding_idx = 0
    embedding_dim, n_mels = 2, 6
    latent_channels = n_mels

    # Encoder
    num_enc_layers = 6

    # Decoder
    hidden_channels = 3
    num_flows, num_dec_layers, num_splits = 3, 2, 2
    down_scale = 2

    # Duration Predictor
    kernel_size = 3

    max_src_length = 20
    max_tgt_length = 2 * max_src_length

    src_length = torch.randint(
        max_src_length // 2, max_src_length + 1, (batch_size,), dtype=torch.long
    )
    tgt_length = torch.randint(
        max_tgt_length // 2, max_tgt_length + 1, (batch_size,), dtype=torch.long
    )
    max_src_length = torch.max(src_length).item()
    max_tgt_length = torch.max(tgt_length).item()
    src = torch.randint(1, num_embeddings, (batch_size, max_src_length))
    tgt = torch.randn((batch_size, n_mels, max_tgt_length))

    encoder = build_encoder(
        num_embeddings,
        embedding_dim,
        latent_channels,
        padding_idx=padding_idx,
        num_layers=num_enc_layers,
    )
    decoder = build_decoder(
        n_mels,
        hidden_channels,
        num_flows=num_flows,
        num_layers=num_dec_layers,
        num_splits=num_splits,
        down_scale=down_scale,
    )
    duration_predictor = FastSpeechDurationPredictor(
        [embedding_dim, 2],
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    length_regulator = LengthRegulator(batch_first=batch_first)

    model = GlowTTS(encoder, decoder, duration_predictor, length_regulator)
    latent, log_duration, padding_mask, logdet = model(
        src,
        tgt,
        src_length=src_length,
        tgt_length=tgt_length,
    )
    src_latent, tgt_latent = latent
    log_est_duration, log_ml_duration = log_duration
    src_padding_mask, tgt_padding_mask = padding_mask

    assert src_latent.size() == (batch_size, max_src_length, embedding_dim)
    assert tgt_latent.size() == (batch_size, max_tgt_length, latent_channels)
    assert log_est_duration.size() == (batch_size, max_src_length)
    assert log_ml_duration.size() == (batch_size, max_src_length)
    assert src_padding_mask.size() == (batch_size, max_src_length)
    assert tgt_padding_mask.size() == (batch_size, max_tgt_length)
    assert logdet.size() == (batch_size,)

    output, est_duration = model.inference(
        src,
        src_length=src_length,
    )

    assert output.size()[:2] == (batch_size, n_mels)
    assert est_duration.size() == (batch_size, max_src_length)


def test_glowtts_encoder() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_embeddings = 5
    padding_idx = 0
    embedding_dim, latent_channels = 2, 4
    max_length = 20

    # Encoder
    num_layers = 6

    length = torch.randint(max_length // 2, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length).item()
    input = torch.randint(1, num_embeddings, (batch_size, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask, padding_idx)

    model = build_encoder(
        num_embeddings,
        embedding_dim,
        latent_channels,
        padding_idx=padding_idx,
        num_layers=num_layers,
    )
    output, normal = model(input)

    z = torch.randn((batch_size, max_length, latent_channels))
    log_prob = normal.log_prob(z)

    assert output.size() == (batch_size, max_length, embedding_dim)
    assert normal.mean.size() == (batch_size, max_length, latent_channels)
    assert normal.stddev.size() == (batch_size, max_length, latent_channels)
    assert log_prob.size() == (batch_size, max_length)

    # FFTrBlock
    d_model = 16
    hidden_channels = 2
    num_heads = 4
    kernel_size = 3
    num_layers = 3

    input = torch.randint(1, num_embeddings, (batch_size, max_length))
    input = input.masked_fill(padding_mask, 0)

    embedding = nn.Embedding(
        num_embeddings,
        d_model,
        padding_idx=padding_idx,
    )
    layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=True,
    )
    backbone = GlowTTSTransformerEncoder(
        layer,
        num_layers=num_layers,
        batch_first=True,
    )
    proj_mean = nn.Linear(d_model, latent_channels)
    proj_std = nn.Linear(d_model, latent_channels)

    model = TextEncoder(
        embedding,
        backbone,
        proj_mean=proj_mean,
        proj_std=proj_std,
    )

    output, normal = model(input)

    z = torch.randn((batch_size, max_length, latent_channels))
    log_prob = normal.log_prob(z)

    assert output.size() == (batch_size, max_length, d_model)
    assert normal.mean.size() == (batch_size, max_length, latent_channels)
    assert normal.stddev.size() == (batch_size, max_length, latent_channels)
    assert log_prob.size() == (batch_size, max_length)


def test_glowtts_decoder() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels, hidden_channels = 4, 3
    num_flows, num_layers, num_splits = 3, 2, 2
    down_scale = 3
    max_length = 20

    model = build_decoder(
        in_channels,
        hidden_channels,
        num_flows=num_flows,
        num_layers=num_layers,
        num_splits=num_splits,
        down_scale=down_scale,
    )

    length = torch.randint(down_scale + 1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, in_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    expanded_padding_mask = padding_mask.unsqueeze(dim=1)

    input = input.masked_fill(expanded_padding_mask, 0)
    z, padding_mask = model(input, padding_mask=padding_mask)
    output, padding_mask = model(z, padding_mask=padding_mask, reverse=True)

    expanded_padding_mask = padding_mask.unsqueeze(dim=1)
    masked_input = input.masked_fill(expanded_padding_mask, 0)

    assert output.size() == input.size()
    allclose(output, masked_input, atol=1e-6)

    zeros = torch.zeros((batch_size,))

    z, padding_mask, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, padding_mask, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    expanded_padding_mask = padding_mask.unsqueeze(dim=1)
    masked_input = input.masked_fill(expanded_padding_mask, 0)

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, masked_input, atol=1e-6)
    allclose(logdet, zeros, atol=1e-4)


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_glowtts_transformer_encoder(batch_first: bool) -> None:
    torch.manual_seed(0)

    batch_size, max_length = 4, 8

    # FFTrBlock
    d_model = 16
    hidden_channels = 2
    num_heads = 2
    kernel_size = 3
    num_layers = 3

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length).item()
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randn((batch_size, max_length, d_model))
    input = input.masked_fill(src_key_padding_mask.unsqueeze(dim=-1), 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    model = GlowTTSTransformerEncoder(
        layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    output = model(input, src_key_padding_mask=src_key_padding_mask)

    assert output.size() == input.size()


def test_glowtts_glow_block() -> None:
    torch.manual_seed(0)

    batch_size = 2
    in_channels, hidden_channels = 8, 6
    num_layers = 4
    max_length = 16

    # w/ 2D padding mask
    model = GlowBlock(in_channels, hidden_channels, num_layers=num_layers)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, in_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input, atol=1e-6)

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
    allclose(output, input, atol=1e-6)
    allclose(logdet, zeros)

    # w/o padding mask
    batch_size = 2
    in_channels, hidden_channels = 8, 6
    max_length = 16

    model = GlowBlock(in_channels, hidden_channels, num_layers=num_layers)

    input = torch.randn(batch_size, in_channels, max_length)

    z = model(input)
    output = model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input, atol=1e-6)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input, atol=1e-6)
    allclose(logdet, zeros)


def build_encoder(
    num_embeddings: int,
    d_model: int,
    latent_channels: int,
    padding_idx: int = 0,
    num_layers: int = 6,
) -> TextEncoder:
    embedding = nn.Embedding(
        num_embeddings,
        d_model,
        padding_idx=padding_idx,
    )

    encoder_layer = FFTrBlock(
        d_model,
        d_model,
        num_heads=2,
        kernel_size=[3, 3],
        batch_first=True,
    )
    backbone = GlowTTSTransformerEncoder(encoder_layer, num_layers=num_layers)
    proj_mean = nn.Linear(d_model, latent_channels)
    proj_std = nn.Linear(d_model, latent_channels)

    model = TextEncoder(
        embedding,
        backbone,
        proj_mean=proj_mean,
        proj_std=proj_std,
    )

    return model


def build_decoder(
    in_channels: int,
    hidden_channels: int,
    num_flows: int,
    num_layers: int,
    num_splits: int = 2,
    down_scale: int = 2,
) -> Decoder:
    model = Decoder(
        in_channels,
        hidden_channels,
        num_flows=num_flows,
        num_layers=num_layers,
        num_splits=num_splits,
        down_scale=down_scale,
    )

    return model


def build_length_regulator(
    num_features: List[int],
    kernel_size: int = 3,
    batch_first: bool = True,
) -> LengthRegulator:
    duration_predictor = FastSpeechDurationPredictor(
        num_features,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    length_regulator = LengthRegulator(duration_predictor, batch_first=batch_first)

    return length_regulator
