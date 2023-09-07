import torch
import torch.nn as nn


class MSELoss(nn.Module):
    def __init__(self, target: float, reduction: str = "mean") -> None:
        super().__init__()

        self.target = target
        self.reduction = reduction

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        loss = (input - self.target) ** 2

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction != "none":
            raise ValueError("Invalid reduction.")

        return loss
