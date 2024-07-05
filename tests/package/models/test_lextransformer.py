import pytest
import torch

from audyn.models.lextransformer import (
    LEXTransformerDecoder,
    LEXTransformerDecoderLayer,
    LEXTransformerEncoder,
    LEXTransformerEncoderLayer,
)

expected_warning_message = (
    "audyn.modeles.lextransformer.{class_name} is deprecated."
    " Use audyn.modules.lextransformer.{class_name} instead."
)


def test_legacy_lextransformer_encoder_layer() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    xpos_base = 100

    with pytest.warns(DeprecationWarning) as e:
        _ = LEXTransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            xpos_base=xpos_base,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="LEXTransformerEncoderLayer"
    )


def test_legacy_lextransformer_decoder_layer() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    xpos_base = 100

    with pytest.warns(DeprecationWarning) as e:
        _ = LEXTransformerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
            xpos_base=xpos_base,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="LEXTransformerDecoderLayer"
    )


def test_legacy_lextransformer_encoder() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5
    xpos_base = 100

    with pytest.warns(DeprecationWarning) as e:
        _ = LEXTransformerEncoder(
            d_model,
            nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            xpos_base=xpos_base,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="LEXTransformerEncoder"
    )


def test_legacy_lextransformer_decoder() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5
    xpos_base = 100

    with pytest.warns(DeprecationWarning) as e:
        _ = LEXTransformerDecoder(
            d_model,
            nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            xpos_base=xpos_base,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="LEXTransformerDecoder"
    )
