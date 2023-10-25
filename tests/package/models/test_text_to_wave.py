import os

import hydra
import torch
from omegaconf import OmegaConf

conf_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "conf")
conf_dir = os.path.join(conf_root, "text_to_wave")


def test_fastspeech_wavenet():
    torch.manual_seed(0)

    config_path = os.path.join(conf_dir, "fastspeech+wavenet.yaml")
    config = OmegaConf.load(config_path)

    batch_size, max_length = 2, 8

    # word embedding
    vocab_size = config.text_to_feat.encoder.word_embedding.num_embeddings
    n_mels = config.text_to_feat.decoder.fc_layer.out_features
    in_channels = config.feat_to_wave.in_channels

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randint(0, vocab_size, (batch_size, max_length))
    input = input.masked_fill(src_key_padding_mask, 0)
    discrete_initial_state = torch.full(
        (batch_size, 1),
        fill_value=in_channels // 2,
        dtype=torch.long,
    )
    continuos_initial_state = torch.zeros(
        (batch_size, in_channels, 1),
        dtype=torch.float,
    )

    model = hydra.utils.instantiate(config)

    output, melspectrogram, log_duration = model.inference(input, discrete_initial_state)

    assert output.size(0) == batch_size
    assert melspectrogram.size()[:2] == (
        batch_size,
        n_mels,
    )
    assert log_duration.size() == (batch_size, max_length)

    output, melspectrogram, log_duration = model.inference(input, continuos_initial_state)

    assert output.size()[:2] == (batch_size, in_channels)
    assert melspectrogram.size()[:2] == (
        batch_size,
        n_mels,
    )
    assert log_duration.size() == (batch_size, max_length)


def test_fastspeech_waveglow():
    torch.manual_seed(0)

    config_path = os.path.join(conf_dir, "fastspeech+waveglow.yaml")
    config = OmegaConf.load(config_path)

    batch_size, max_length = 2, 8

    # word embedding
    vocab_size = config.text_to_feat.encoder.word_embedding.num_embeddings
    n_mels = config.text_to_feat.decoder.fc_layer.out_features

    length = torch.randint(1, max_length, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    src_key_padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = torch.randint(0, vocab_size, (batch_size, max_length))
    input = input.masked_fill(src_key_padding_mask, 0)

    model = hydra.utils.instantiate(config)

    output, melspectrogram, log_duration = model.inference(input)

    assert output.size(0) == batch_size
    assert melspectrogram.size()[:2] == (
        batch_size,
        n_mels,
    )
    assert log_duration.size() == (batch_size, max_length)
