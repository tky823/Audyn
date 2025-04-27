from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

__all__ = [
    "DistanceBasedNegativeSamplingLoss",
]


class DistanceBasedNegativeSamplingLoss(nn.Module):
    """Distance-based negative sampling loss.

    Args:
        distance (nn.Module or callable): Function to compute distance between two points.
            For input (*batch_shape, embedding_dim) and target (*batch_shape, embedding_dim),
            ``distance`` should return distance (*batch_shape,).

    """

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

        print(self.training)

        if self.training:
            positive_distance = self.distance(anchor, positive, **positive_distance_kwargs)

            if positive_distance.size(-1) == 0:
                # corner case: root in DAG
                positive_distance = 0
        else:
            positive_distance = self.distance(
                anchor.unsqueeze(dim=-2), positive, **positive_distance_kwargs
            )

            if positive_distance.size(-1) == 0:
                # corner case: root in DAG
                positive_distance = 0
            else:
                positive_distance = torch.logsumexp(positive_distance, dim=-1)

        negative_distance = self.distance(
            anchor.unsqueeze(dim=-2), negative, **negative_distance_kwargs
        )

        if negative_distance.size(-1) == 0:
            negative_distance = 0
        else:
            negative_distance = torch.logsumexp(-negative_distance, dim=-1)

        loss = positive_distance + negative_distance

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
