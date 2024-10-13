from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "TaggingBCEWithLogitsLoss",
]

evaluation_means = {
    "arithmetic",
    "geometric",
    "none",
}


class TaggingBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    """BCEWithLogitsLoss for tagging.

    Args:
        evaluation_mean (str): How to compute mean during evaluation.
            ``arithmetic``, ``geometric``, and ``none`` are supported.

    This class is useful when

        - Training data loader returns ``(batch_size, num_tags, *)``.
        - Evaluation data loader returns ``(num_segments, num_tags, *)``,
            where num_segments means number of segments extracted from one long sequence.

    """

    def __init__(
        self,
        evaluation_mean: Optional[str] = "arithmetic",
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduce=None,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

        assert evaluation_mean in evaluation_means

        self.evaluation_mean = evaluation_mean

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        evaluation_mean = self.evaluation_mean

        if self.training:
            logit = input
        else:
            if evaluation_mean == "arithmetic":
                input = F.sigmoid(input)
                input = input.mean(dim=0, keepdim=True)
                logit = torch.logit(input)

                if target.dim() == input.dim():
                    target = target.mean(dim=0, keepdim=True)
                elif target.dim() + 1 == input.dim():
                    target = target.unsqueeze(dim=0)
                else:
                    raise ValueError("Invalid shape of target is given.")
            elif evaluation_mean == "geometric":
                logit = input.mean(dim=0, keepdim=True)

                if target.dim() == input.dim():
                    target = target.mean(dim=0, keepdim=True)
                elif target.dim() + 1 == input.dim():
                    target = target.unsqueeze(dim=0)
                else:
                    raise ValueError("Invalid shape of target is given.")
            elif evaluation_mean == "none":
                logit = input
            else:
                raise ValueError(f"{evaluation_mean} is not supported as evaluation_mean.")

        loss = super().forward(logit, target)

        return loss
