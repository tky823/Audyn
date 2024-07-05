import pytest
import torch

from audyn.models.roformer import (
    RoFormerDecoder,
    RoFormerDecoderLayer,
    RoFormerEncoder,
    RoFormerEncoderLayer,
)

expected_warning_message = (
    "audyn.modeles.roformer.{class_name} is deprecated."
    " Use audyn.modules.roformer.{class_name} instead."
)


def test_legacy_roformer_encoder_layer() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3

    with pytest.warns(DeprecationWarning) as e:
        _ = RoFormerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="RoFormerEncoderLayer"
    )


def test_legacy_roformer_decoder_layer() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3

    with pytest.warns(DeprecationWarning) as e:
        _ = RoFormerDecoderLayer(
            d_model,
            nhead,
            dim_feedforward,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(
        class_name="RoFormerDecoderLayer"
    )


def test_legacy_roformer_encoder() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5

    with pytest.warns(DeprecationWarning) as e:
        _ = RoFormerEncoder(
            d_model,
            nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(class_name="RoFormerEncoder")


def test_legacy_roformer_decoder() -> None:
    torch.manual_seed(0)

    d_model, nhead, dim_feedforward = 8, 2, 3
    num_layers = 5

    with pytest.warns(DeprecationWarning) as e:
        _ = RoFormerDecoder(
            d_model,
            nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
        )

    assert len(e.list) > 0
    assert str(e.list[-1].message) == expected_warning_message.format(class_name="RoFormerDecoder")
