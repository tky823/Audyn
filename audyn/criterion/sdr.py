from typing import Optional

import torch
import torch.nn as nn

from .pit import PIT

__all__ = [
    "SISDR",
    "NegSISDR",
    "PITNegSISDR",
]


class SISDR(nn.Module):
    """SI-SDR.

    See https://arxiv.org/abs/1811.02508 for the details.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()

        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = _sisdr(input, target, eps=self.eps)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction type {self.reduction} is given.")

        return loss


class NegSISDR(nn.Module):
    """Negative SI-SDR.

    See https://arxiv.org/abs/1811.02508 for the details.
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()

        self.reduction = reduction

        if reduction not in ["mean", "sum", "none"]:
            raise ValueError("Invalid reduction type")

        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = -_sisdr(input, target, eps=self.eps)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction type {self.reduction} is given.")

        return loss


class PITNegSISDR(PIT):
    """negative SI-SDR for permutation invariant training."""

    def __init__(
        self, reduction: str = "mean", num_sources: Optional[int] = None, eps: float = 1e-8
    ) -> None:
        criterion = NegSISDR(reduction="none", eps=eps)

        super().__init__(criterion, num_sources=num_sources)

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _ = super().forward(input, target)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction type {self.reduction} is given.")

        return loss


def _sisdr(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    r"""Sample-wise SI-SDR.

    Args:
        input (torch.Tensor): Noisy waveform of shape (\*, length).
        target (torch.Tensor): Target clean waveform (\*, length).

    Returns:
        torch.Tensor: Sample-wise SI-SDR of shape (\*,).

    """
    alpha = torch.sum(input * target, dim=-1, keepdim=True) / (
        torch.sum(target**2, dim=-1, keepdim=True) + eps
    )
    loss = (torch.sum((alpha * target) ** 2, dim=-1) + eps) / (
        torch.sum((alpha * target - input) ** 2, dim=-1) + eps
    )
    loss = 10 * torch.log10(loss)

    return loss
