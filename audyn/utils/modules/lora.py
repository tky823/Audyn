import copy

import torch.nn as nn

from ...modules.lora import LoRALinear, LoRAMultiheadAttention

__all__ = [
    "apply_lora",
    "apply_lora_to_linear",
    "apply_lora_to_mha",
]


def apply_lora(module: nn.Module, rank: int = 8, persistent: bool = False) -> nn.Module:
    """Apply LoRA to modules. Now, only ``nn.Linear`` and ``nn.MultiheadAttention`` are supported.

    Args:
        module (nn.Module): Module to which LoRA applies.
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

    Returns:
        nn.Module: Applied module.

    """
    if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
        module = copy.deepcopy(module)
    elif isinstance(module, nn.Linear):
        module = apply_lora_to_linear(
            module,
            rank=rank,
            persistent=persistent,
        )
    elif isinstance(module, nn.MultiheadAttention):
        module = apply_lora_to_mha(
            module,
            rank=rank,
            persistent=persistent,
        )
    else:
        for name, child_module in module.named_children():
            child_module = apply_lora(
                child_module,
                rank=rank,
                persistent=persistent,
            )

            setattr(module, name, child_module)

    return module


def apply_lora_to_linear(module: nn.Linear, rank: int = 8, persistent: bool = False) -> LoRALinear:
    weight = module.weight.data

    if module.bias is None:
        bias = None
    else:
        bias = module.bias.data

    factory_kwargs = {
        "device": weight.device,
        "dtype": weight.dtype,
    }

    applied = LoRALinear(
        weight,
        bias=bias,
        rank=rank,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied


def apply_lora_to_mha(
    module: nn.MultiheadAttention, rank: int = 8, persistent: bool = False
) -> LoRAMultiheadAttention:
    weight = module.in_proj_weight

    factory_kwargs = {
        "device": weight.device,
        "dtype": weight.dtype,
    }

    applied = LoRAMultiheadAttention(
        module.num_heads,
        weight,
        in_proj_bias=module.in_proj_bias,
        bias_k=module.bias_k,
        bias_v=module.bias_v,
        add_zero_attn=module.add_zero_attn,
        dropout=module.dropout,
        out_proj_weight=module.out_proj.weight,
        out_proj_bias=module.out_proj.bias,
        q_proj_weight=module.q_proj_weight,
        k_proj_weight=module.k_proj_weight,
        v_proj_weight=module.v_proj_weight,
        batch_first=module.batch_first,
        rank=rank,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied
