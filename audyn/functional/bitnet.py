from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "bitlinear158",
    "round_clip",
    "round_clamp",
]


def bitlinear158(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """BitLinear158 function."""
    # quantize input
    q = 2 ** (bits - 1)
    abs_input = torch.abs(input)
    gamma = torch.max(abs_input)
    gamma = torch.clamp(gamma, min=eps)
    x = input * q / gamma
    x = round_clip(x, min=-q, max=q - 1)

    # quantize weight
    abs_weight = torch.abs(weight)
    beta = torch.mean(abs_weight)
    beta = torch.clamp(beta, min=eps)
    weight = abs_weight / beta
    quantized_weight = round_clip(weight, min=-1, max=1)

    x = F.linear(x, quantized_weight, bias=bias)
    output = x * (beta * gamma) / q

    return output


def round_clip(
    input: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    """Differntiable round + clip used in BitNet.

    .. note::

        Gradient is given by straight through estimator.

    """
    kwargs = {}

    if min is not None:
        kwargs["min"] = min

    if max is not None:
        kwargs["max"] = max

    x = torch.round(input)

    if len(kwargs) > 0:
        x = torch.clamp(x, **kwargs)

    output = torch.detach(x - input) + input

    return output


def round_clamp(
    input: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    """Alias of audyn.functional.bitnet.round_clip."""
    return round_clip(input, min=min, max=max)
