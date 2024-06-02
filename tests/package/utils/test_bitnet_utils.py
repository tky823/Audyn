import torch
import torch.nn as nn

from audyn.modules.bitnet import BitLinear158, BitMultiheadAttention158
from audyn.utils.model.bitnet import convert_to_bitlinear158


def test_convert_to_bitlinear158() -> None:
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
        BitLinear158(10, 12, bias=True),
        BitLinear158(12, 5, bias=False),
        nn.Sequential(
            BitLinear158(5, 3, bias=True),
            BitLinear158(3, 2, bias=False),
        ),
    )

    model = convert_to_bitlinear158(model, bits=8)

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
        BitLinear158(10, 12, bias=True),
        BitMultiheadAttention158(12, 4, bias=False),
        nn.Sequential(
            BitMultiheadAttention158(12, 3, bias=True),
            BitLinear158(3, 2, bias=False),
        ),
    )

    model = convert_to_bitlinear158(model, bits=8)

    # for extra_repr
    print(model)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()
