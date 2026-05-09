import copy
import warnings
from typing import Optional

import torch.nn as nn

from ...modules.activation import (
    ExtrapolatablePositionalMultiheadAttention,
    PartialRotaryPositionalMultiheadAttention,
    RotaryPositionalMultiheadAttention,
)
from ...modules.lora import (
    LoRAExtrapolatablePositionalMultiheadAttention,
    LoRALinear,
    LoRAMultiheadAttention,
    LoRAPartialRotaryPositionalMultiheadAttention,
    LoRARotaryPositionalMultiheadAttention,
)

__all__ = [
    "apply_lora",
    "apply_lora_to_linear",
    "apply_lora_to_mha",
]


def apply_lora(
    module: nn.Module,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
) -> nn.Module:
    """Apply LoRA to modules. Now, only ``nn.Linear`` and ``nn.MultiheadAttention`` are supported.

    Args:
        module (nn.Module): Module to which LoRA applies.
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        alpha (float): Scaling factor that controls magnitude of LoRA update. Update
            is scaled by ``alpha / rank``. Default: ``rank``.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

    Returns:
        nn.Module: Applied module.

    """
    if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
        module = copy.deepcopy(module)
    elif isinstance(module, nn.Linear):
        if type(module) is nn.Linear:
            module = apply_lora_to_linear(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )
        else:
            warnings.warn(f"{type(module)} does not support LoRA.", stacklevel=2)

            module = copy.deepcopy(module)
    elif isinstance(module, nn.MultiheadAttention):
        if type(module) is RotaryPositionalMultiheadAttention:
            module = apply_lora_to_rope_mha(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )
        elif type(module) is ExtrapolatablePositionalMultiheadAttention:
            module = apply_lora_to_xpos_mha(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )
        elif type(module) is PartialRotaryPositionalMultiheadAttention:
            module = apply_lora_to_partial_rope_mha(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )
        elif type(module) is nn.MultiheadAttention:
            module = apply_lora_to_mha(
                module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )
        else:
            warnings.warn(f"{type(module)} does not support LoRA.", stacklevel=2)

            module = copy.deepcopy(module)
    else:
        for name, child_module in module.named_children():
            child_module = apply_lora(
                child_module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                persistent=persistent,
            )

            setattr(module, name, child_module)

    return module


def apply_lora_to_linear(
    module: nn.Linear,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
) -> LoRALinear:
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
        alpha=alpha,
        dropout=dropout,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied


def apply_lora_to_mha(
    module: nn.MultiheadAttention,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
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
        alpha=alpha,
        lora_dropout=dropout,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied


def apply_lora_to_rope_mha(
    module: RotaryPositionalMultiheadAttention,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
) -> LoRARotaryPositionalMultiheadAttention:
    weight = module.in_proj_weight

    factory_kwargs = {
        "device": weight.device,
        "dtype": weight.dtype,
    }

    applied = LoRARotaryPositionalMultiheadAttention(
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
        base=module.rope.base,
        share_heads=module.share_heads,
        batch_first=module.batch_first,
        rank=rank,
        alpha=alpha,
        lora_dropout=dropout,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied


def apply_lora_to_xpos_mha(
    module: ExtrapolatablePositionalMultiheadAttention,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
) -> LoRAExtrapolatablePositionalMultiheadAttention:
    weight = module.in_proj_weight

    factory_kwargs = {
        "device": weight.device,
        "dtype": weight.dtype,
    }

    applied = LoRAExtrapolatablePositionalMultiheadAttention(
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
        base=module.k_xpos.base,
        share_heads=module.share_heads,
        batch_first=module.batch_first,
        rank=rank,
        alpha=alpha,
        lora_dropout=dropout,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied


def apply_lora_to_partial_rope_mha(
    module: PartialRotaryPositionalMultiheadAttention,
    rank: int = 8,
    alpha: Optional[float] = None,
    dropout: float = 0.05,
    persistent: bool = False,
) -> LoRAPartialRotaryPositionalMultiheadAttention:
    weight = module.in_proj_weight

    factory_kwargs = {
        "device": weight.device,
        "dtype": weight.dtype,
    }

    applied = LoRAPartialRotaryPositionalMultiheadAttention(
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
        base=module.rope.base,
        share_heads=module.share_heads,
        fraction=module.fraction,
        batch_first=module.batch_first,
        rank=rank,
        alpha=alpha,
        lora_dropout=dropout,
        persistent=persistent,
        **factory_kwargs,
    )

    return applied
