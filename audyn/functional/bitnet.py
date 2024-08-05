from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

__all__ = [
    "bitlinear158",
    "bitlinear158_inference",
    "round_clip",
    "round_clamp",
    "quantize_weight",
]


def bitlinear158(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """BitLinear158 function."""
    q = 2 ** (bits - 1)
    quantized_input, gamma = quantize_input(input, dim=dim, bits=bits, eps=eps)
    quantized_weight, scale = quantize_weight(weight, eps=eps)
    x = F.linear(quantized_input, quantized_weight, bias=bias)
    output = x * (scale * gamma) / q

    return output


def bitlinear158_inference(
    input: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
) -> torch.Tensor:
    """BitLinear158 function for inference.

    Unlike ``bitlinear158``, ``bitlinear158_inference`` takes
    input, quantized weight, scale, and optional bias.
    """
    q = 2 ** (bits - 1)
    quantized_input, gamma = quantize_input(input, dim=dim, bits=bits, eps=eps)
    x = F.linear(quantized_input, quantized_weight, bias=bias)
    output = x * (scale * gamma) / q

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

    if torch.is_grad_enabled():
        output = x
    else:
        output = torch.detach(x - input) + input

    return output


def round_clamp(
    input: torch.Tensor,
    min: Optional[float] = None,
    max: Optional[float] = None,
) -> torch.Tensor:
    """Alias of audyn.functional.bitnet.round_clip."""
    return round_clip(input, min=min, max=max)


def quantize_input(
    input: torch.Tensor,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    q = 2 ** (bits - 1)
    abs_input = torch.abs(input)

    if dim is None:
        gamma = torch.max(abs_input)
    else:
        # compute max per group
        # e.g. When abs_input.size() is (4, 10, 3, 5)
        #      and dim is (1, 3), gamma.size()
        #      becomes (4, 1, 3, 1).
        n_dims = abs_input.dim()
        dims = list(range(n_dims))

        if isinstance(dim, int):
            group_dims = []

            if dim < 0:
                dim = n_dims + dim

            group_dim = dims.pop(dim)
            group_dims.append(group_dim)
            batch_dims = dims
        else:
            group_dims = []

            for _dim in dim:
                if _dim < 0:
                    _dim = n_dims + _dim

                group_dims.append(_dim)

            group_dims = sorted(group_dims)

            for _dim in group_dims[::-1]:
                dims.pop(_dim)

            batch_dims = dims

        dims = batch_dims + group_dims
        abs_input = abs_input.permute(*dims).contiguous()

        start_dim = len(batch_dims)
        flattened_abs_input = torch.flatten(abs_input, start_dim=start_dim)
        gamma, _ = torch.max(flattened_abs_input, dim=-1)

        for _dim in group_dims:
            gamma = gamma.unsqueeze(dim=_dim)

    gamma = torch.clamp(gamma, min=eps)
    x = input * q / gamma
    output = round_clip(x, min=-q, max=q - 1)

    return output, gamma


def quantize_weight(
    weight: torch.Tensor,
    eps: float = 1e-5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    abs_weight = torch.abs(weight)
    scale = torch.mean(abs_weight)
    scale = torch.clamp(scale, min=eps)
    weight = weight / scale
    quantized_weight = round_clip(weight, min=-1, max=1)

    return quantized_weight, scale
