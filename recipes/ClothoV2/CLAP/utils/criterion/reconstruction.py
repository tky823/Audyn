from typing import Optional

import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, length: Optional[torch.LongTensor] = None
    ) -> torch.Tensor:
        """Forward pass of ReconstructionLoss.

        Args:
            input (torch.Tensor): Feature of shape (batch_size, max_length, num_features).
            target (torch.Tensor): Feature of shape (batch_size, max_length, num_features).

        Returns:
            torch.Tensor: Computed loss.

        """
        reduction = self.reduction

        batch_size, max_length, _ = input.size()

        factory_kwargs = {
            "dtype": torch.long,
            "device": input.device,
        }

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)

        loss = (input - target) ** 2
        loss = loss.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
        loss = loss.sum(dim=-2) / length.unsqueeze(dim=-1)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
