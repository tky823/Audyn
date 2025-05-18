import torch
import torch.nn as nn


class CustomCriterion(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = target - input

        return torch.mean(loss)
