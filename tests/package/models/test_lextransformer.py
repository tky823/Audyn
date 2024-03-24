import pytest
import torch
from dummy import allclose

from audyn.models.lextransformer import (
    LEXTransformerDecoder,
    LEXTransformerDecoderLayer,
    LEXTransformerEncoder,
    LEXTransformerEncoderLayer,
)


@pytest.mark.parametrize("batch_first", [True, False])
def test_lextransformer_encoder_layer(batch_first: bool) -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    batch_size, src_length = 4, 16
    xpos_base = 100
    shift_size = src_length

    model = LEXTransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        xpos_base=xpos_base,
        batch_first=batch_first,
    )

    # to deactivate dropout
    model.eval()

    if batch_first:
        src = torch.randn((batch_size, src_length, d_model))
    else:
        src = torch.randn((src_length, batch_size, d_model))

    output = model(src)

    if batch_first:
        random_padding = torch.randn((batch_size, shift_size, d_model))
        padded_src = torch.cat([random_padding, src], dim=1)
    else:
        random_padding = torch.randn((shift_size, batch_size, d_model))
        padded_src = torch.cat([random_padding, src], dim=0)

    src_key_padding_mask = torch.arange(src_length + shift_size) < shift_size
    src_key_padding_mask = src_key_padding_mask.expand((batch_size, src_length + shift_size))

    padded_output = model(padded_src, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        _, padded_output = torch.split(padded_output, [shift_size, src_length], dim=1)
    else:
        _, padded_output = torch.split(padded_output, [shift_size, src_length], dim=0)

    allclose(output, padded_output, atol=1e-5)


@pytest.mark.parametrize("batch_first", [True, False])
def test_lextransformer_decoder_layer(batch_first: bool) -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    batch_size, tgt_length, memory_length = 4, 16, 20
    xpos_base = 100
    shift_size = tgt_length

    model = LEXTransformerDecoderLayer(
        d_model,
        nhead,
        dim_feedforward,
        xpos_base=xpos_base,
        batch_first=batch_first,
    )

    # to deactivate dropout
    model.eval()

    if batch_first:
        tgt = torch.randn((batch_size, tgt_length, d_model))
        memory = torch.randn((batch_size, memory_length, d_model))
    else:
        tgt = torch.randn((tgt_length, batch_size, d_model))
        memory = torch.randn((memory_length, batch_size, d_model))

    output = model(tgt, memory)

    if batch_first:
        random_tgt_padding = torch.randn((batch_size, shift_size, d_model))
        random_memory_padding = torch.randn((batch_size, shift_size, d_model))
        padded_tgt = torch.cat([random_tgt_padding, tgt], dim=1)
        padded_memory = torch.cat([random_memory_padding, memory], dim=1)
    else:
        random_tgt_padding = torch.randn((shift_size, batch_size, d_model))
        random_memory_padding = torch.randn((shift_size, batch_size, d_model))
        padded_tgt = torch.cat([random_tgt_padding, tgt], dim=0)
        padded_memory = torch.cat([random_memory_padding, memory], dim=0)

    tgt_key_padding_mask = torch.arange(tgt_length + shift_size) < shift_size
    memory_key_padding_mask = torch.arange(memory_length + shift_size) < shift_size
    tgt_key_padding_mask = tgt_key_padding_mask.expand((batch_size, tgt_length + shift_size))
    memory_key_padding_mask = memory_key_padding_mask.expand(
        (batch_size, memory_length + shift_size)
    )

    padded_output = model(
        padded_tgt,
        padded_memory,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    if batch_first:
        _, padded_output = torch.split(padded_output, [shift_size, tgt_length], dim=1)
    else:
        _, padded_output = torch.split(padded_output, [shift_size, tgt_length], dim=0)

    allclose(output, padded_output, atol=1e-5)


@pytest.mark.parametrize("batch_first", [True, False])
def test_lextransformer_encoder(batch_first: bool) -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5
    batch_size, src_length = 4, 16
    xpos_base = 100
    shift_size = src_length

    model = LEXTransformerEncoder(
        d_model,
        nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        xpos_base=xpos_base,
        batch_first=batch_first,
    )

    # to deactivate dropout
    model.eval()

    if batch_first:
        src = torch.randn((batch_size, src_length, d_model))
    else:
        src = torch.randn((src_length, batch_size, d_model))

    output = model(src)

    if batch_first:
        random_padding = torch.randn((batch_size, shift_size, d_model))
        padded_src = torch.cat([random_padding, src], dim=1)
    else:
        random_padding = torch.randn((shift_size, batch_size, d_model))
        padded_src = torch.cat([random_padding, src], dim=0)

    src_key_padding_mask = torch.arange(src_length + shift_size) < shift_size
    src_key_padding_mask = src_key_padding_mask.expand((batch_size, src_length + shift_size))

    padded_output = model(padded_src, src_key_padding_mask=src_key_padding_mask)

    if batch_first:
        _, padded_output = torch.split(padded_output, [shift_size, src_length], dim=1)
    else:
        _, padded_output = torch.split(padded_output, [shift_size, src_length], dim=0)

    allclose(output, padded_output, atol=1e-3)


@pytest.mark.parametrize("batch_first", [True, False])
def test_lextransformer_decoder(batch_first: bool) -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5
    batch_size, tgt_length, memory_length = 4, 16, 20
    xpos_base = 100
    shift_size = tgt_length

    model = LEXTransformerDecoder(
        d_model,
        nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        xpos_base=xpos_base,
        batch_first=batch_first,
    )

    # to deactivate dropout
    model.eval()

    if batch_first:
        tgt = torch.randn((batch_size, tgt_length, d_model))
        memory = torch.randn((batch_size, memory_length, d_model))
    else:
        tgt = torch.randn((tgt_length, batch_size, d_model))
        memory = torch.randn((memory_length, batch_size, d_model))

    output = model(tgt, memory)

    if batch_first:
        random_tgt_padding = torch.randn((batch_size, shift_size, d_model))
        random_memory_padding = torch.randn((batch_size, shift_size, d_model))
        padded_tgt = torch.cat([random_tgt_padding, tgt], dim=1)
        padded_memory = torch.cat([random_memory_padding, memory], dim=1)
    else:
        random_tgt_padding = torch.randn((shift_size, batch_size, d_model))
        random_memory_padding = torch.randn((shift_size, batch_size, d_model))
        padded_tgt = torch.cat([random_tgt_padding, tgt], dim=0)
        padded_memory = torch.cat([random_memory_padding, memory], dim=0)

    tgt_key_padding_mask = torch.arange(tgt_length + shift_size) < shift_size
    memory_key_padding_mask = torch.arange(memory_length + shift_size) < shift_size
    tgt_key_padding_mask = tgt_key_padding_mask.expand((batch_size, tgt_length + shift_size))
    memory_key_padding_mask = memory_key_padding_mask.expand(
        (batch_size, memory_length + shift_size)
    )

    padded_output = model(
        padded_tgt,
        padded_memory,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
    )

    if batch_first:
        _, padded_output = torch.split(padded_output, [shift_size, tgt_length], dim=1)
    else:
        _, padded_output = torch.split(padded_output, [shift_size, tgt_length], dim=0)

    allclose(output, padded_output, atol=1e-5)
