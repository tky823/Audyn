import pytest
import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.modules.lora import LoRALinear
from audyn.utils.modules.lora import apply_lora


@pytest.mark.parametrize("persistent", [True, False])
def test_appy_lora(persistent: bool) -> None:
    torch.manual_seed(0)

    rank = 2

    model = nn.Sequential(
        nn.Linear(10, 12, bias=True),
        nn.Linear(12, 5, bias=False),
        nn.Sequential(
            nn.Linear(5, 3, bias=True),
            nn.Linear(3, 2, bias=False),
        ),
    )
    expected_model = nn.Sequential(
        LoRALinear.build_from_linear(
            nn.Linear(10, 12, bias=True), rank=rank, persistent=persistent
        ),
        LoRALinear.build_from_linear(
            nn.Linear(12, 5, bias=False), rank=rank, persistent=persistent
        ),
        nn.Sequential(
            LoRALinear.build_from_linear(
                nn.Linear(5, 3, bias=True), rank=rank, persistent=persistent
            ),
            LoRALinear.build_from_linear(
                nn.Linear(3, 2, bias=False), rank=rank, persistent=persistent
            ),
        ),
    )

    model = apply_lora(model, rank=rank, persistent=persistent)

    # for extra_repr
    print(model)

    for p, p_expected in zip(model.parameters(), expected_model.parameters()):
        assert p.size() == p_expected.size()


@pytest.mark.parametrize("batch_first", [True, False])
def test_appy_lora_to_transformer_encoder(batch_first: bool) -> None:
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
    lora_model = apply_lora(model)

    input = torch.randn((length, batch_size, d_model))

    if batch_first:
        input = input.transpose(1, 0)

    model.eval()
    lora_model.eval()

    output = model(input)
    lora_output = lora_model(input)

    allclose(lora_output, output)
