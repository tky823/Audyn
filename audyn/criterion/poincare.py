import torch
import torch.nn as nn

from ..functional.poincare import poincare_distance

__all__ = [
    "PoincareDistanceLoss",
]


class PoincareDistanceLoss(nn.Module):
    """Distance between two points on Poincare ball."""

    def __init__(
        self,
        curvature: float = -1,
        dim: int = -1,
        reduction: str = "mean",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        self.curvature = curvature
        self.dim = dim
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of PoincareDistanceLoss.

        Args:
            input (torch.Tensor): Input point of shape (batch_shape, embedding_dim)
                when ``dim=-1``.
            target (torch.Tensor): Target point of shape (batch_shape, embedding_dim)
                when ``dim=-1``.

        Returns:
            torch.Tensor: Computed loss.

        """
        curvature = self.curvature
        dim = self.dim
        reduction = self.reduction
        eps = self.eps

        loss = poincare_distance(input, target, curvature=curvature, dim=dim, eps=eps)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
