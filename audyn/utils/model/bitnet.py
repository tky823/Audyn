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
    remove_bias: bool = False,
) -> nn.Module:
    """Convert nn.Linear to BitLinearB158 in given module.

    Args:
        module (nn.Module): Module to be converted.
        bits (int): Number of quantization bits in scale parameter.
        eps (float): Tiny value to avoid zero division.
        remove_bias (bool): If ``True``, bias is forced to be removed.

    Returns:
        nn.Module: Converted module.

    """
    if isinstance(module, nn.Linear):
        module = convert_linear_to_bit_linear_158(
            module,
            bits,
            eps=eps,
            remove_bias=remove_bias,
        )
    else:
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.Linear):
                converted = convert_linear_to_bit_linear_158(
                    child_module,
                    bits=bits,
                    eps=eps,
                    remove_bias=remove_bias,
                )
            else:
                converted = convert_to_bit_linear_158(
                    child_module, bits=bits, eps=eps, remove_bias=remove_bias
                )

            setattr(module, name, converted)

    return module


def convert_linear_to_bit_linear_158(
    module: nn.Linear,
    bits: int = 8,
    eps: float = 1e-5,
    remove_bias: bool = False,
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

    if bias and not remove_bias:
        converted.bias.data.copy_(module.bias.data)

    return converted
