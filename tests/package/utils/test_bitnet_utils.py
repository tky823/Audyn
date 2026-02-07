import pytest
import torch
import torch.nn as nn

from audyn.modules.bitnet import (
    BitLinear158,
    BitLinear158Inference,
    BitMultiheadAttention158,
    BitMultiheadAttention158Inference,
)
from audyn.utils.modules.bitnet import convert_to_bitlinear158, convert_to_bitlinear158_inference


@pytest.mark.parametrize("remove_bias", [True, False])
def test_convert_to_bitlinear158(remove_bias: bool) -> None:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 12, bias=True),
        nn.Linear(12, 5, bias=False),
        nn.Sequential(
            nn.Linear(5, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        ),
    )
    expected_model = nn.Sequential(
        BitLinear158(10, 12, bias=True and not remove_bias),
        BitLinear158(12, 5, bias=False),
        nn.Sequential(
            BitLinear158(5, 3, bias=True and not remove_bias),
            BitLinear158(3, 2, bias=False),
        ),
    )

    model = convert_to_bitlinear158(model, bits=8, remove_bias=remove_bias)

    # for extra_repr
    print(model)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()

    model = nn.Sequential(
        nn.Linear(10, 12, bias=True),
        nn.MultiheadAttention(12, 4, bias=False),
        nn.Sequential(
            nn.MultiheadAttention(12, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        ),
    )
    expected_model = nn.Sequential(
        BitLinear158(10, 12, bias=True and not remove_bias),
        BitMultiheadAttention158(12, 4, bias=False),
        nn.Sequential(
            BitMultiheadAttention158(12, 3, bias=True and not remove_bias),
            BitLinear158(3, 2, bias=False),
        ),
    )

    model = convert_to_bitlinear158(model, bits=8, remove_bias=remove_bias)

    # for extra_repr
    print(model)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()


@pytest.mark.parametrize("remove_bias", [True, False])
def test_convert_to_bitlinear158_inference(remove_bias: bool) -> None:
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(10, 12, bias=True),
        nn.Linear(12, 5, bias=False),
        nn.Sequential(
            nn.Linear(5, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        ),
    )
    expected_model = nn.Sequential(
        BitLinear158Inference.build_from_bitlinear158(
            BitLinear158(10, 12, bias=True and not remove_bias)
        ),
        BitLinear158Inference.build_from_bitlinear158(BitLinear158(12, 5, bias=False)),
        nn.Sequential(
            BitLinear158Inference.build_from_bitlinear158(
                BitLinear158(5, 3, bias=True and not remove_bias),
            ),
            BitLinear158Inference.build_from_bitlinear158(
                BitLinear158(3, 2, bias=False),
            ),
        ),
    )

    model = convert_to_bitlinear158_inference(model, bits=8, remove_bias=remove_bias)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()

    model = nn.Sequential(
        nn.Linear(10, 12, bias=True),
        nn.MultiheadAttention(12, 4, bias=False),
        nn.Sequential(
            nn.MultiheadAttention(12, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        ),
    )
    expected_model = nn.Sequential(
        BitLinear158Inference.build_from_bitlinear158(
            BitLinear158(10, 12, bias=True and not remove_bias)
        ),
        BitMultiheadAttention158Inference.build_from_bitmha158(
            BitMultiheadAttention158(12, 4, bias=False)
        ),
        nn.Sequential(
            BitMultiheadAttention158Inference.build_from_bitmha158(
                BitMultiheadAttention158(12, 3, bias=True and not remove_bias)
            ),
            BitLinear158Inference.build_from_bitlinear158(BitLinear158(3, 2, bias=False)),
        ),
    )

    model = convert_to_bitlinear158_inference(model, bits=8, remove_bias=remove_bias)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()


@pytest.mark.parametrize("remove_bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_convert_to_bitlinear158_transformer_encoder(remove_bias: bool, batch_first: bool) -> None:
    torch.manual_seed(0)

    d_model, dim_feedforward = 24, 16
    nhead = 4
    batch_size = 5
    length = 10

    model = nn.TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        batch_first=batch_first,
    )
    model = convert_to_bitlinear158(model, bits=8, remove_bias=remove_bias)

    input = torch.randn((length, batch_size, d_model))

    if batch_first:
        input = input.transpose(1, 0)

    output = model(input)

    if batch_first:
        assert output.size() == (batch_size, length, d_model)
    else:
        assert output.size() == (length, batch_size, d_model)

    model = nn.TransformerEncoderLayer(
        d_model,
        nhead,
        dim_feedforward=dim_feedforward,
        batch_first=batch_first,
    )
    model = convert_to_bitlinear158_inference(model, bits=8, remove_bias=remove_bias)

    input = torch.randn((length, batch_size, d_model))

    if batch_first:
        input = input.transpose(1, 0)

    output = model(input)

    if batch_first:
        assert output.size() == (batch_size, length, d_model)
    else:
        assert output.size() == (length, batch_size, d_model)
