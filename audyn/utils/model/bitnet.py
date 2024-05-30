import torch.nn as nn

from ...modules.bitnet import BitLinearB158

__all__ = [
    "convert_to_bit_linear_158",
    "convert_linear_to_bit_linear_158",
]


def convert_to_bit_linear_158(
    module: nn.Module,
    bits: int = 8,
    eps: float = 1e-5,
) -> nn.Module:
    """Convert nn.Linear to BitLinearB158 in given module."""
    if isinstance(module, nn.Linear):
        module = convert_linear_to_bit_linear_158(module, bits, eps=eps)
    else:
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear):
                converted = convert_linear_to_bit_linear_158(child_module, bits, eps=eps)
            else:
                converted = convert_to_bit_linear_158(child_module, bits, eps=eps)

            setattr(module, name, converted)

    return module


def convert_linear_to_bit_linear_158(
    module: nn.Linear,
    bits: int = 8,
    eps: float = 1e-5,
) -> BitLinearB158:
    """Convert nn.Linear to BitLinearB158."""
    factory_kwargs = {
        "device": module.weight.device,
        "dtype": module.weight.dtype,
    }

    if module.bias is None:
        bias = False
    else:
        bias = True

    in_features = module.in_features
    out_features = module.out_features

    converted = BitLinearB158(
        in_features,
        out_features,
        bits=bits,
        bias=bias,
        eps=eps,
        **factory_kwargs,
    )
    converted.weight.data.copy_(module.weight.data)

    if module.bias is not None:
        converted.bias.data.copy_(module.bias.data)

    return converted
