import math

import torch
import torch.nn as nn

__all__ = ["NLLLoss", "GaussFlowLoss"]


class NLLLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        logdet: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of general flow loss.

        This loss just reverse the sign of input log-determinant.

        Args:
            logdet (torch.Tensor): Log-determinant of shape (batch_size,).

        Returns:
            torch.Tensor: Negative log-likelihood.

        """
        reduction = self.reduction

        loss = -logdet

        if reduction == "mean":
            loss = loss.mean(dim=0)
        elif reduction == "sum":
            loss = loss.sum(dim=0)
        else:
            raise ValueError("reduction = {} is not supported.".format(reduction))

        return loss


class GaussFlowLoss(nn.Module):
    def __init__(self, std: float = 1, reduction: str = "mean") -> None:
        super().__init__()

        self.std = std
        self.reduction = reduction

    def forward(
        self,
        input: torch.Tensor,
        logdet: torch.Tensor = 0,
    ) -> torch.Tensor:
        """Forward pass of Gaussian-based flow loss.

        Args:
            input (torch.Tensor): Probability variable of shape (batch_size, *).
            logdet (torch.Tensor): Log-determinant of shape (batch_size,).

        Returns:
            torch.Tensor: Negative log-likelihood.

        """
        reduction = self.reduction

        z = input

        n_dims = z.dim()
        dims = tuple(range(1, n_dims))
        z_2 = torch.sum(z * z, dim=dims)
        dim = z.numel() / z.size(0)

        nll_gauss = math.log(2 * math.pi) / 2 + math.log(self.std) + z_2 / (2 * dim * self.std**2)
        loss = nll_gauss - logdet / dim

        if reduction == "mean":
            loss = loss.mean(dim=0)
        elif reduction == "sum":
            loss = loss.sum(dim=0)
        else:
            raise ValueError("reduction = {} is not supported.".format(reduction))

        return loss
