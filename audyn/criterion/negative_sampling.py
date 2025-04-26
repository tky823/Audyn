from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "DistanceBasedNegativeSamplingLoss",
]


class DistanceBasedNegativeSamplingLoss(nn.Module):

    def __init__(
        self,
        distance: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
        reduction: str = "mean",
        positive_distance_kwargs: Optional[Dict[str, Any]] = None,
        negative_distance_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if positive_distance_kwargs is None:
            positive_distance_kwargs = {}

        if negative_distance_kwargs is None:
            negative_distance_kwargs = {}

        self.distance = distance
        self.reduction = reduction
        self.positive_distance_kwargs = positive_distance_kwargs
        self.negative_distance_kwargs = negative_distance_kwargs

    def forward(
        self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of NegativeSamplingLoss.

        Args:
            anchor (torch.Tensor): (*, embedding_dim).
            positive (torch.Tensor): (*, embedding_dim).
            negative (torch.Tensor): (*, num_neg_samples, embedding_dim).

        Returns:
            torch.Tensor: Computed loss.

        """
        reduction = self.reduction
        positive_distance_kwargs = self.positive_distance_kwargs
        negative_distance_kwargs = self.negative_distance_kwargs

        positive_distance = self.distance(anchor, positive, **positive_distance_kwargs)
        negative_distance = self.distance(
            anchor.unsqueeze(dim=-2), negative, **negative_distance_kwargs
        )

        loss = positive_distance + torch.logsumexp(-negative_distance, dim=-1)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
