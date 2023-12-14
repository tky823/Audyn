import pytest
import torch
import torch.nn as nn

from audyn.models.fastspeech import (
    Decoder,
    Encoder,
    FastSpeech,
    LengthRegulator,
    MultiSpeakerFastSpeech,
)
from audyn.modules.duration_predictor import FastSpeechDurationPredictor
from audyn.modules.fastspeech import FFTrBlock
from audyn.modules.positional_encoding import AbsolutePositionalEncoding
from audyn.utils.duration import transform_log_duration

parameters_batch_first = [True, False]


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_fastspeech(batch_first: bool):
    torch.manual_seed(0)

    batch_size, max_length = 2, 8

    # word embedding
    vocab_size = 4
    padding_idx = 0
    d_model = 16

    # FFTrBlock
    hidden_channels = 2
    num_heads = 2
    kernel_size = 3

    # Encoder & Decoder
    out_features = 4
    num_layers = 3

    # LengthRegulator
    num_features = [d_model, 2]

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randint(0, vocab_size, (batch_size, max_length))
    input = input.masked_fill(src_key_padding_mask, 0)
    duration = torch.randint(1, 3, (batch_size, max_length), dtype=torch.long)
    duration = duration.masked_fill(src_key_padding_mask, 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    word_embedding = nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
    )
    positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
    layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    encoder = Encoder(
        word_embedding,
        positional_encoding,
        encoder_layer=layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    fc_layer = nn.Linear(d_model, out_features)
    decoder = Decoder(
        positional_encoding,
        decoder_layer=layer,
        fc_layer=fc_layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    duration_predictor = FastSpeechDurationPredictor(
        num_features, kernel_size=kernel_size, batch_first=batch_first
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    model = FastSpeech(
        encoder,
        decoder,
        duration_predictor,
        length_regulator,
        batch_first=batch_first,
    )

    # w/o duration
    output, log_est_duration = model(input)

    assert output.size(2) == out_features
    assert log_est_duration.size()[:2] == (batch_size, max_length)

    if batch_first:
        assert output.size(0) == batch_size
    else:
        assert output.size(1) == batch_size

    # w/ duration
    output, log_est_duration = model(input, duration=duration)

    assert output.size(2) == out_features
    assert log_est_duration.size()[:2] == (batch_size, max_length)

    if batch_first:
        assert output.size(0) == batch_size
    else:
        assert output.size(1) == batch_size


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_multispk_fastspeech(batch_first: bool):
    torch.manual_seed(0)

    batch_size, max_length = 2, 8

    # word embedding
    vocab_size = 4
    padding_idx = 0
    d_model = 16

    # FFTrBlock
    hidden_channels = 2
    num_heads = 2
    kernel_size = 3

    # Encoder & Decoder
    out_features = 4
    num_layers = 3

    # LengthRegulator
    num_features = [d_model, 2]

    # SpeakerEmbedding
    num_speakers = 4

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randint(0, vocab_size, (batch_size, max_length), dtype=torch.long)
    input = input.masked_fill(src_key_padding_mask, 0)
    speaker = torch.randint(0, num_speakers, (batch_size,), dtype=torch.long)
    duration = torch.randint(1, 3, (batch_size, max_length), dtype=torch.long)
    duration = duration.masked_fill(src_key_padding_mask, 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    word_embedding = nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
    )
    positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
    layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    encoder = Encoder(
        word_embedding,
        positional_encoding,
        encoder_layer=layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    fc_layer = nn.Linear(d_model, out_features)
    decoder = Decoder(
        positional_encoding,
        decoder_layer=layer,
        fc_layer=fc_layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    duration_predictor = FastSpeechDurationPredictor(
        num_features, kernel_size=kernel_size, batch_first=batch_first
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    speaker_encoder = nn.Embedding(
        num_embeddings=num_speakers, embedding_dim=d_model, padding_idx=padding_idx
    )
    model = MultiSpeakerFastSpeech(
        encoder,
        decoder,
        duration_predictor,
        length_regulator,
        speaker_encoder=speaker_encoder,
        batch_first=batch_first,
    )

    # w/o duration
    output, log_est_duration = model(input, speaker=speaker)

    assert output.size(2) == out_features
    assert log_est_duration.size()[:2] == (batch_size, max_length)

    if batch_first:
        assert output.size(0) == batch_size
    else:
        assert output.size(1) == batch_size

    # w/ duration
    output, log_est_duration = model(input, speaker=speaker, duration=duration)

    assert output.size(2) == out_features
    assert log_est_duration.size()[:2] == (batch_size, max_length)

    if batch_first:
        assert output.size(0) == batch_size
    else:
        assert output.size(1) == batch_size


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_fastspeech_encoder(batch_first: bool):
    torch.manual_seed(0)

    batch_size, max_length = 2, 8

    # word embedding
    vocab_size = 4
    padding_idx = 0
    d_model = 16

    # FFTrBlock
    hidden_channels = 2
    num_heads = 2
    kernel_size = 3

    # Encoder
    num_layers = 3

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randint(0, vocab_size, (batch_size, max_length))
    input = input.masked_fill(src_key_padding_mask, 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    word_embedding = nn.Embedding(
        num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=padding_idx
    )
    positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
    encoder_layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    encoder = Encoder(
        word_embedding,
        positional_encoding,
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    output = encoder(input, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        assert output.size() == (batch_size, max_length, d_model)
    else:
        assert output.size() == (max_length, batch_size, d_model)

    encoder_layer = nn.TransformerEncoderLayer(
        d_model,
        nhead=num_heads,
        dim_feedforward=hidden_channels,
        batch_first=batch_first,
    )
    encoder = Encoder(
        word_embedding,
        positional_encoding,
        encoder_layer=encoder_layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )
    output = encoder(input, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        assert output.size() == (batch_size, max_length, d_model)
    else:
        assert output.size() == (max_length, batch_size, d_model)


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_fastspeech_decoder(batch_first: bool):
    torch.manual_seed(0)

    batch_size, max_length = 2, 8

    # FFTrBlock
    d_model, hidden_channels, out_features = 16, 2, 3
    num_heads = 2
    kernel_size = 3

    # Decoder
    num_layers = 3

    positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
    decoder_layer = FFTrBlock(
        d_model,
        hidden_channels,
        num_heads=num_heads,
        kernel_size=kernel_size,
        batch_first=batch_first,
    )
    fc_layer = nn.Linear(d_model, out_features)
    decoder = Decoder(
        positional_encoding,
        decoder_layer=decoder_layer,
        fc_layer=fc_layer,
        num_layers=num_layers,
        batch_first=batch_first,
    )

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    tgt_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randn((batch_size, max_length, d_model))
    input = input.masked_fill(tgt_key_padding_mask.unsqueeze(dim=-1), 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    output = decoder(input, tgt_key_padding_mask=tgt_key_padding_mask)

    if batch_first:
        assert output.size() == (batch_size, max_length, out_features)
    else:
        assert output.size() == (max_length, batch_size, out_features)


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_length_regulator(batch_first: bool):
    torch.manual_seed(0)

    batch_size, max_length = 2, 8
    num_features = [4, 2]
    kernel_size = 3

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randn((batch_size, max_length, num_features[0]))
    input = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

    if not batch_first:
        input = input.swapaxes(1, 0)

    duration_predictor = FastSpeechDurationPredictor(
        num_features, kernel_size=kernel_size, batch_first=batch_first
    )
    length_regulator = LengthRegulator(batch_first=batch_first)
    log_est_diration = duration_predictor(input)
    linear_est_diration = transform_log_duration(log_est_diration)
    output, log_est_duration = length_regulator(
        input, linear_est_diration, padding_mask=padding_mask
    )

    if batch_first:
        assert output.size(0) == batch_size
    else:
        assert output.size(1) == batch_size

    assert output.size(-1) == num_features[0]
    assert log_est_duration.size() == (batch_size, max_length)
