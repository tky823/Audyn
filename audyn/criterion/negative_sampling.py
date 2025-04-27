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
            positive (torch.Tensor): Positive embeddings of shape (*, embedding_dim) if
                ``self.training=True``. Otherwise, (*, num_pos_samples, embedding_dim).
            negative (torch.Tensor): Negative embeddings of shape
                (*, num_neg_samples, embedding_dim).

        Returns:
            torch.Tensor: Computed loss.

        .. note::

            We assume shape of positive depends on ``self.training``.

        """
        reduction = self.reduction
        positive_distance_kwargs = self.positive_distance_kwargs
        negative_distance_kwargs = self.negative_distance_kwargs

        negative_distance = self.distance(
            anchor.unsqueeze(dim=-2), negative, **negative_distance_kwargs
        )

        if self.training:
            positive_distance = self.distance(anchor, positive, **positive_distance_kwargs)

            if positive_distance.dim() > 0 and positive_distance.size(-1) == 0:
                raise ValueError("Positive sample is required during training.")

            _positive_distance = positive_distance.unsqueeze(dim=-1)
            distances = torch.cat([_positive_distance, negative_distance], dim=-1)
        else:
            positive_distance = self.distance(
                anchor.unsqueeze(dim=-2), positive, **positive_distance_kwargs
            )

            # corner case: root in DAG
            if positive_distance.dim() > 0 and positive_distance.size(-1) == 0:
                positive_distance = None

            if negative_distance.dim() > 0 and negative_distance.size(-1) == 0:
                negative_distance = None

            distances = []

            if positive_distance is not None:
                distances.append(positive_distance)
                positive_distance = torch.logsumexp(positive_distance, dim=-1)

            if negative_distance is not None:
                distances.append(negative_distance)

            distances = torch.cat(distances, dim=-1)

        distances = torch.logsumexp(-distances, dim=-1)

        loss = distances

        if positive_distance is not None:
            loss = loss + positive_distance

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"{reduction} is not supported as reduction.")

        return loss
