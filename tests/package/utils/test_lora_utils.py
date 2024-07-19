import pytest
import torch.nn as nn

from audyn.modules.lora import LoRALinear
from audyn.utils.modules.lora import apply_lora


@pytest.mark.parametrize("persistent", [True, False])
def test_lora_linear(persistent: bool) -> None:
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
