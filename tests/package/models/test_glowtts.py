import os
import tempfile
from typing import List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from dummy import allclose
from torch.distributions import Independent
from torch.distributions.normal import Normal

from audyn.models.fastspeech import LengthRegulator
from audyn.models.glowtts import (
    Decoder,
    GlowBlock,
    GlowTTS,
    GlowTTSTransformerEncoder,
    PreNet,
    TextEncoder,
)
from audyn.models.text_to_wave import FastSpeechWaveNetBridge
from audyn.modules.duration_predictor import FastSpeechDurationPredictor
from audyn.modules.fastspeech import FFTrBlock
from audyn.modules.glowtts import GlowTTSFFTrBlock

parameters_batch_first = [True, False]


@pytest.mark.parametrize("scaling", [True, False])
@pytest.mark.parametrize("channel_dependent_scaling", [True, False])
def test_glowtts(scaling: bool, channel_dependent_scaling: bool) -> None:
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

    if channel_dependent_scaling:
        scaling_channels = (n_mels * down_scale) // 2
    else:
        scaling_channels = None

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
        scaling=scaling,
        scaling_channels=scaling_channels,
    )
    duration_predictor = FastSpeechDurationPredictor(
        [embedding_dim, 2],
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    transform_middle = FastSpeechWaveNetBridge(take_exp=False)

    model = GlowTTS(
        encoder,
        decoder,
        duration_predictor,
        length_regulator,
        transform_middle=transform_middle,
    )
    latent, log_duration, padding_mask, logdet = model(
        src,
        tgt,
        src_length=src_length,
        tgt_length=tgt_length,
    )
    src_latent, tgt_latent = latent
    log_est_duration, ml_duration = log_duration
    src_padding_mask, tgt_padding_mask = padding_mask

    assert src_latent.size() == (batch_size, max_src_length, embedding_dim)
    assert tgt_latent.size() == (batch_size, max_tgt_length, latent_channels)
    assert log_est_duration.size() == (batch_size, max_src_length)
    assert ml_duration.size() == (batch_size, max_src_length)
    assert src_padding_mask.size() == (batch_size, max_src_length)
    assert tgt_padding_mask.size() == (batch_size, max_tgt_length)
    assert logdet.size() == (batch_size,)

    output, est_duration = model.inference(
        src,
        src_length=src_length,
    )

    assert output.size()[:2] == (batch_size, n_mels)
    assert est_duration.size() == (batch_size, max_src_length)

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "model.pth")

        state_dict = model.state_dict()
        torch.save(state_dict, path)

        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)

    # test search_gaussian_monotonic_alignment
    input = torch.randn((batch_size, max_tgt_length, n_mels))
    mean = torch.randn((batch_size, max_src_length, n_mels))
    log_std = torch.randn((batch_size, max_src_length, n_mels))
    log_prob_tuple, ml_duration_tuple = GlowTTS.search_gaussian_monotonic_alignment(
        input, (mean, log_std)
    )

    normal = Normal(loc=mean, scale=torch.exp(log_std))
    normal = Independent(normal, reinterpreted_batch_ndims=1)
    log_prob_normal, ml_duration_normal = GlowTTS.search_gaussian_monotonic_alignment(
        input, normal
    )
    assert torch.allclose(log_prob_tuple, log_prob_normal, atol=1e-5)
    assert torch.equal(ml_duration_tuple, ml_duration_normal)


def test_glowtts_unbatched() -> None:
    """Ensure output of GlowTTS does not depend on padding length."""
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
    scaling_channels = None

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
        scaling=False,
        scaling_channels=scaling_channels,
    )
    duration_predictor = FastSpeechDurationPredictor(
        [embedding_dim, 2],
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    transform_middle = FastSpeechWaveNetBridge(take_exp=False)

    model = GlowTTS(
        encoder,
        decoder,
        duration_predictor,
        length_regulator,
        transform_middle=transform_middle,
    )
    model.eval()

    latent, log_duration, padding_mask, logdet = model(
        src,
        tgt,
        src_length=src_length,
        tgt_length=tgt_length,
    )
    src_latent, tgt_latent = latent
    log_est_duration, ml_duration = log_duration
    src_padding_mask, tgt_padding_mask = padding_mask

    unbatched_src, _ = torch.split(src, [1, batch_size - 1], dim=0)
    unbatched_tgt, _ = torch.split(tgt, [1, batch_size - 1], dim=0)
    unbatched_src_length, _ = torch.split(src_length, [1, batch_size - 1], dim=0)
    unbatched_tgt_length, _ = torch.split(tgt_length, [1, batch_size - 1], dim=0)

    # pad value for test
    unbatched_src = F.pad(unbatched_src, (0, 1))
    unbatched_tgt = F.pad(unbatched_tgt, (0, 1))

    unbatched_latent, unbatched_log_duration, unbatched_padding_mask, unbatched_logdet = model(
        unbatched_src,
        unbatched_tgt,
        src_length=unbatched_src_length,
        tgt_length=unbatched_tgt_length,
    )

    unbatched_src_latent, unbatched_tgt_latent = unbatched_latent
    unbatched_log_est_duration, unbatched_ml_duration = unbatched_log_duration
    unbatched_src_padding_mask, unbatched_tgt_padding_mask = unbatched_padding_mask

    unbatched_src_latent = unbatched_src_latent.squeeze(dim=0)
    unbatched_tgt_latent = unbatched_tgt_latent.squeeze(dim=0)
    unbatched_log_est_duration = unbatched_log_est_duration.squeeze(dim=0)
    unbatched_ml_duration = unbatched_ml_duration.squeeze(dim=0)
    unbatched_src_padding_mask = unbatched_src_padding_mask.squeeze(dim=0)
    unbatched_tgt_padding_mask = unbatched_tgt_padding_mask.squeeze(dim=0)
    unbatched_logdet = unbatched_logdet.squeeze(dim=0)

    assert torch.allclose(unbatched_src_latent[:-1], src_latent[0])
    assert torch.allclose(unbatched_tgt_latent[:-1], tgt_latent[0])
    assert torch.allclose(unbatched_log_est_duration[:-1], log_est_duration[0])
    assert torch.allclose(unbatched_ml_duration[:-1], ml_duration[0])
    assert torch.allclose(unbatched_src_padding_mask[:-1], src_padding_mask[0])
    assert torch.allclose(unbatched_tgt_padding_mask[:-1], tgt_padding_mask[0])
    assert torch.allclose(unbatched_logdet, logdet[0])


@pytest.mark.parametrize("use_relative_position", [True, False])
def test_official_glowtts(use_relative_position: bool) -> None:
    num_embeddings = 148  # or 149
    n_mels = 80
    batch_size = 4

    # Encoder
    padding_idx = 0
    d_model = 192
    pre_kernel_size = 5
    num_pre_layers = 3
    window_size = 4
    share_heads = True
    enc_hidden_channels = 768
    num_heads = 2
    enc_kernel_size = 3
    enc_dropout = 0.1
    batch_first = True
    num_enc_layers = 6

    # Duration Predictor
    dp_hidden_channels = 256
    dp_kernel_size = 3
    stop_gradient = True

    # Decoder
    dec_hidden_channels = 192
    num_flows = 12
    num_splits = 4
    down_scale = 2
    num_dec_layers = 4
    dec_kernel_size = 5
    dilation_rate = 5
    dec_dropout = 0.05

    max_src_length = 10
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

    embedding = nn.Embedding(
        num_embeddings,
        d_model,
        padding_idx=padding_idx,
    )
    pre_net = PreNet(
        d_model,
        d_model,
        d_model,
        kernel_size=pre_kernel_size,
        batch_first=batch_first,
        num_layers=num_pre_layers,
    )

    if use_relative_position:
        encoder_layer = GlowTTSFFTrBlock(
            d_model,
            enc_hidden_channels,
            num_heads=num_heads,
            kernel_size=[enc_kernel_size, enc_kernel_size],
            dropout=enc_dropout,
            batch_first=batch_first,
            window_size=window_size,
            share_heads=share_heads,
        )
    else:
        encoder_layer = FFTrBlock(
            d_model,
            enc_hidden_channels,
            num_heads=num_heads,
            kernel_size=[enc_kernel_size, enc_kernel_size],
            dropout=enc_dropout,
            batch_first=batch_first,
        )

    backbone = GlowTTSTransformerEncoder(
        encoder_layer,
        num_layers=num_enc_layers,
        batch_first=batch_first,
    )
    proj_mean = nn.Linear(d_model, n_mels)
    proj_std = nn.Linear(d_model, n_mels)

    encoder = TextEncoder(
        embedding,
        backbone,
        proj_mean=proj_mean,
        proj_std=proj_std,
        pre_net=pre_net,
    )
    decoder = Decoder(
        n_mels,
        dec_hidden_channels,
        num_flows=num_flows,
        num_layers=num_dec_layers,
        num_splits=num_splits,
        down_scale=down_scale,
        kernel_size=dec_kernel_size,
        dilation_rate=dilation_rate,
        dropout=dec_dropout,
    )

    duration_predictor = FastSpeechDurationPredictor(
        [d_model, dp_hidden_channels, dp_hidden_channels],
        kernel_size=dp_kernel_size,
        stop_gradient=stop_gradient,
        batch_first=batch_first,
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    transform_middle = FastSpeechWaveNetBridge(take_exp=False)

    model = GlowTTS(
        encoder,
        decoder,
        duration_predictor,
        length_regulator,
        transform_middle=transform_middle,
    )

    num_parameters = 0

    for p in model.parameters():
        if p.requires_grad:
            num_parameters += p.numel()

    # According to paper, number of parameters is 28.6M.
    if use_relative_position:
        assert num_parameters == 28639329
    else:
        assert num_parameters == 28628961

    latent, log_duration, padding_mask, logdet = model(
        src,
        tgt,
        src_length=src_length,
        tgt_length=tgt_length,
    )
    src_latent, tgt_latent = latent
    log_est_duration, log_ml_duration = log_duration
    src_padding_mask, tgt_padding_mask = padding_mask

    assert src_latent.size() == (batch_size, max_src_length, d_model)
    assert tgt_latent.size() == (batch_size, max_tgt_length, n_mels)
    assert log_est_duration.size() == (batch_size, max_src_length)
    assert log_ml_duration.size() == (batch_size, max_src_length)
    assert src_padding_mask.size() == (batch_size, max_src_length)
    assert tgt_padding_mask.size() == (batch_size, max_tgt_length)
    assert logdet.size() == (batch_size,)


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
    allclose(logdet, zeros, atol=1e-6)


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_glowtts_transformer_encoder(batch_first: bool) -> None:
    torch.manual_seed(0)

    batch_size, max_length = 5, 8

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

    # ensure invariance of zero padding
    model.eval()

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length).item()
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randn((batch_size, max_length, d_model))
    input = input.masked_fill(src_key_padding_mask.unsqueeze(dim=-1), 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    output = model(input, src_key_padding_mask=src_key_padding_mask)

    src_key_padding_mask = torch.arange(2 * max_length) >= length.unsqueeze(dim=-1)
    random_padding = torch.randn((batch_size, max_length, d_model))

    if batch_first:
        padded_input = torch.cat([input, random_padding], dim=1)
    else:
        random_padding = random_padding.swapaxes(1, 0)
        padded_input = torch.cat([input, random_padding], dim=0)

    padded_output = model(padded_input, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        padded_output, _ = torch.split(padded_output, [max_length, max_length], dim=1)
    else:
        padded_output, _ = torch.split(padded_output, [max_length, max_length], dim=0)

    allclose(padded_output, output, atol=1e-6)


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
    backbone = GlowTTSTransformerEncoder(
        encoder_layer,
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

    return model


def build_decoder(
    in_channels: int,
    hidden_channels: int,
    num_flows: int,
    num_layers: int,
    num_splits: int = 2,
    down_scale: int = 2,
    scaling: bool = False,
    scaling_channels: Optional[int] = None,
) -> Decoder:
    model = Decoder(
        in_channels,
        hidden_channels,
        num_flows=num_flows,
        num_layers=num_layers,
        num_splits=num_splits,
        down_scale=down_scale,
        scaling=scaling,
        scaling_channels=scaling_channels,
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
