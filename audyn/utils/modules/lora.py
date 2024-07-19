import copy

import torch.nn as nn

from ...modules.lora import LoRALinear

__all__ = [
    "apply_lora",
    "apply_lora_to_linear",
]


def apply_lora(module: nn.Module, rank: int = 8, persistent: bool = False) -> nn.Module:
    """Apply LoRA to modules. Now, only ``nn.Linear`` is supported.

    Args:
        module (nn.Module): Module to which LoRA applies.
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

    Returns:
        nn.Module: Applied module.

    """
    if isinstance(module, LoRALinear):
        module = copy.deepcopy(module)
    elif isinstance(module, nn.Linear):
        module = apply_lora_to_linear(
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
