import pytest
import torch
import torch.nn as nn
from audyn_test import allclose

from audyn.modules.lora import LoRALinear, LoRAMultiheadAttention


@pytest.mark.parametrize("persistent", [True, False])
def test_lora_linear(persistent: bool) -> None:
    batch_size = 3
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

    input = torch.randn((batch_size, in_features))
    output = linear(input)
    lora_output = lora_linear(input)

    assert lora_output.size() == output.size()
    allclose(lora_output, output)

    lora_linear = LoRALinear.build_from_linear(linear, rank=rank, persistent=persistent)
    lora_output = lora_linear(input)

    assert lora_output.size() == output.size()
    allclose(lora_output, output)


@pytest.mark.parametrize("batch_first", [True, False])
@pytest.mark.parametrize("persistent", [True, False])
def test_lora_mha(batch_first: bool, persistent: bool) -> None:
    batch_size = 3
    length = 10
    embed_dim, num_heads = 32, 4
    rank = 2

    mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
    lora_mha = LoRAMultiheadAttention(
        mha.num_heads,
        mha.in_proj_weight,
        in_proj_bias=mha.in_proj_bias,
        bias_k=mha.bias_k,
        bias_v=mha.bias_v,
        add_zero_attn=mha.add_zero_attn,
        dropout=mha.dropout,
        out_proj_weight=mha.out_proj.weight,
        out_proj_bias=mha.out_proj.bias,
        q_proj_weight=mha.q_proj_weight,
        k_proj_weight=mha.k_proj_weight,
        v_proj_weight=mha.v_proj_weight,
        batch_first=mha.batch_first,
        rank=rank,
        persistent=persistent,
    )

    state_dict_keys = set(lora_mha.state_dict().keys())

    if persistent:
        assert state_dict_keys == {
            "in_proj_weight",
            "in_proj_bias",
            "in_proj_weight_in",
            "q_proj_weight_out",
            "k_proj_weight_out",
            "v_proj_weight_out",
            "out_proj.weight",
            "out_proj.bias",
            "out_proj.weight_in",
            "out_proj.weight_out",
        }
    else:
        assert state_dict_keys == {
            "in_proj_weight_in",
            "q_proj_weight_out",
            "k_proj_weight_out",
            "v_proj_weight_out",
            "out_proj.weight_in",
            "out_proj.weight_out",
        }

    input = torch.randn((length, batch_size, embed_dim))

    if batch_first:
        input = input.transpose(1, 0)

    output, attn_weights = mha(input, input, input)
    lora_output, loar_attn_weights = lora_mha(input, input, input)

    assert lora_output.size() == output.size()
    assert attn_weights.size() == loar_attn_weights.size()
    allclose(lora_output, output, atol=1e-6)
    allclose(attn_weights, loar_attn_weights, atol=1e-6)

    lora_mha = LoRAMultiheadAttention.build_from_mha(mha, rank=rank, persistent=persistent)
    lora_output, loar_attn_weights = lora_mha(input, input, input)

    assert lora_output.size() == output.size()
    assert attn_weights.size() == loar_attn_weights.size()
    allclose(lora_output, output, atol=1e-6)
    allclose(attn_weights, loar_attn_weights, atol=1e-6)
