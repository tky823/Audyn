import torch

from .hyperbolic import mobius_add


def poincare_distance(
    input: torch.Tensor,
    other: torch.Tensor,
    curvature: float = -1,
    dim: int = -1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Distance between two points on Poincare ball with negative curvature."""
    assert dim == -1

    _curvature = (-curvature) ** 0.5
    distance = mobius_add(-input, other, curvature=curvature, eps=eps)
    norm = _curvature * torch.linalg.vector_norm(distance, dim=dim)
    scale = 2 / _curvature
    norm = torch.clamp(norm, max=1 - eps)
    output = scale * torch.atanh(norm)

    return output
