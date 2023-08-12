import torch
import torch.nn as nn

__all__ = ["AutoRegressiveWrapper"]


class AutoRegressiveWrapper(nn.Module):
    """Wrapper class for loss function of autoregressive models.

    Args:
        criterion (nn.Module): Criterion.
        dim (int): Dimension of temporal shift.

    """

    def __init__(self, criterion: nn.Module, dim: int = -1) -> None:
        super().__init__()

        self.criterion = criterion
        self.dim = dim

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        dim = self.dim
        shifted_input, _ = torch.split(input, [input.size(dim) - 1, 1], dim=dim)
        _, shifted_target = torch.split(target, [1, target.size(dim) - 1], dim=dim)
        loss = self.criterion(shifted_input, shifted_target, *args, **kwargs)

        return loss
