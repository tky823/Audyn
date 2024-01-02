import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GANCriterion", "DiscriminatorHingeLoss"]


class GANCriterion:
    """Base class of criterion for GAN."""

    def __init__(self, generator: nn.Module, discriminator: nn.Module) -> None:
        self.generator = generator
        self.discriminator = discriminator


class DiscriminatorHingeLoss(nn.Module):
    def __init__(self, minimize: bool, margin: float = 1, reduction: str = "mean") -> None:
        """

        Args:
            minimize (bool): Whether to promote input to minimization or not.

        """
        super().__init__()

        self.minimize = minimize
        self.margin = margin
        self.reduction = reduction

        assert margin >= 0, "margin should be non-negative."

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        minimize = self.minimize
        margin = self.margin
        reduction = self.reduction

        if minimize:
            loss = F.relu(margin + input)
        else:
            loss = F.relu(margin - input)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction != "none":
            raise ValueError(f"reduction={reduction} is not supported.")

        return loss
