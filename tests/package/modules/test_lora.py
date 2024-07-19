import pytest
import torch.nn as nn

from audyn.modules.lora import LoRALinear


@pytest.mark.parametrize("persistent", [True, False])
def test_lora_linear(persistent: bool) -> None:
    in_features, out_features = 32, 16
    rank = 2

    linear = nn.Linear(in_features, out_features)
    lora_linear = LoRALinear(
        linear.weight,
        bias=linear.bias,
        rank=rank,
        persistent=persistent,
    )

    state_dict_keys = set(lora_linear.state_dict().keys())

    if persistent:
        assert state_dict_keys == {"weight", "bias", "weight_in", "weight_out"}
    else:
        assert state_dict_keys == {"weight_in", "weight_out"}
