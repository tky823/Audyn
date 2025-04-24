from typing import Union

import torch

__all__ = [
    "mobius_add",
    "mobius_sub",
    "mobius_scaler_mul",
]


def mobius_add(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: Union[float, torch.Tensor] = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply Mobius addition.

    Args:
        input (torch.Tensor): Vectors of shape (*, num_features).
        other (torch.Tensor): Vectors of shape (*, num_features).
        curvature (float or torch.Tensor): Secctional curvature. Default: ``-1``.

    Returns:
        torch.Tensor: Vectors of shape (*, num_features).

    """
    assert dim == -1, "Only dim=-1 is supported."

    if not isinstance(input, torch.Tensor):
        if isinstance(other, torch.Tensor):
            factory_kwargs = {
                "dtype": other.dtype,
                "device": other.device,
            }

        input = torch.tensor(input, **factory_kwargs)

    if not isinstance(other, torch.Tensor):
        if isinstance(input, torch.Tensor):
            factory_kwargs = {
                "dtype": input.dtype,
                "device": input.device,
            }

        other = torch.tensor(other, **factory_kwargs)

    target_shape = torch.broadcast_shapes(input.size(), other.size())
    input = input.expand(target_shape).contiguous()
    other = other.expand(target_shape).contiguous()

    *batch_shape, num_features = input.size()

    input = input.view(-1, num_features)
    other = other.view(-1, num_features)

    dot = torch.sum(input * other, dim=-1)
    norm_input = torch.sum(input**2, dim=-1)
    norm_other = torch.sum(other**2, dim=-1)

    coeff_input = 1 - 2 * curvature * dot - curvature * norm_other
    coeff_other = 1 + curvature * norm_input
    denom = 1 - 2 * curvature * dot + (curvature**2) * norm_input * norm_other
    num = coeff_input.unsqueeze(dim=-1) * input + coeff_other.unsqueeze(dim=-1) * other
    denom = torch.clamp(denom, min=eps)
    output = num / denom.unsqueeze(dim=-1)
    output = output.view(*batch_shape, num_features)

    return output


def mobius_sub(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: Union[float, torch.Tensor] = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply Mobius subtraction.

    Args:
        input (torch.Tensor): Vectors of shape (*, num_features).
        other (torch.Tensor): Vectors of shape (*, num_features).
        curvature (float or torch.Tensor): Negative curvature.

    Returns:
        torch.Tensor: Vectors of shape (*, num_features).

    """
    return mobius_add(input, -other, curvature=curvature, dim=dim, eps=eps)


def mobius_scaler_mul(
    input: torch.Tensor,
    scalar: Union[float, torch.Tensor],
    curvature: Union[float, torch.Tensor] = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Apply scaler Mobius multiplication.

    Args:
        input (torch.Tensor): Vectors of shape (*, num_features).
        scalar (float or torch.Tensor): Float scaler or scalers of shape (*,).
        curvature (float or torch.Tensor): Secctional curvature. Default: ``-1``.

    Returns:
        torch.Tensor: Vectors of shape (*, num_features).

    """
    assert dim == -1, "Only dim=-1 is supported."

    norm = torch.linalg.vector_norm(input, dim=-1)
    norm = torch.clamp(norm, min=eps)
    denom = ((-curvature) ** 0.5) * norm
    normalized_input = input / denom.unsqueeze(dim=-1)
    tanh = ((-curvature) ** 0.5) * norm
    tanh = torch.clamp(tanh, min=-1 + eps, max=1 - eps)
    scale = torch.tanh(scalar * torch.atanh(tanh))
    output = normalized_input * scale.unsqueeze(dim=-1)

    return output
