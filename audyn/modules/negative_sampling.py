from typing import Any, Optional

import torch
import torch.nn as nn

__all__ = [
    "NegativeSamplingModel",
]


class NegativeSamplingModel(nn.Module):
    """Wrapper of nn.Embedding for negative sampling.

    Args:
        embedding (nn.Embedding): Embedding module.

    """

    def __init__(
        self,
        embedding: nn.Embedding,
        anchor_kwargs: Optional[dict[str, Any]] = None,
        positive_kwargs: Optional[dict[str, Any]] = None,
        negative_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if anchor_kwargs is None:
            anchor_kwargs = {}

        if positive_kwargs is None:
            positive_kwargs = {}

        if negative_kwargs is None:
            negative_kwargs = {}

        self.embedding = embedding
        self.anchor_kwargs = anchor_kwargs
        self.positive_kwargs = positive_kwargs
        self.negative_kwargs = negative_kwargs

    def forward(
        self,
        anchor: torch.LongTensor,
        positive: torch.LongTensor,
        negative: torch.LongTensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of NegativeSamplingModel.

        Args:
            anchor (torch.LongTensor): Anchor indices.
            positive (torch.LongTensor): Positive indices.
            negative (torch.LongTensor): Negative indices.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Anchor embeddings.
                - torch.Tensor: Positive embeddings.
                - torch.Tensor: Negative embeddings.

        """
        anchor_kwargs = self.anchor_kwargs
        positive_kwargs = self.positive_kwargs
        negative_kwargs = self.negative_kwargs

        anchor = self.embedding(anchor, *args, **kwargs, **anchor_kwargs)
        positive = self.embedding(positive, *args, **kwargs, **positive_kwargs)
        negative = self.embedding(negative, *args, **kwargs, **negative_kwargs)

        return anchor, positive, negative
