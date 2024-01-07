from typing import Optional

import torch
import torch.nn as nn


class VALLECrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -1,
        reduction: str = "mean",
        label_smoothing: float = 0,
    ) -> None:
        assert ignore_index == -1

        super().__init__(
            weight=weight,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )

    def forward(self, input: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """Forward pass of VALLECrossEntropyLoss.

        Args:
            input (torch.Tensor): Logit of estimated codebook indices of
                shape (batch_size, length, codebook_size + 1), where additional class means
                EOS token of codebooks.
            target (torch.LongTensor): Groundtruth codebook indices of shape (batch_size, length).
                Padding values should be 0.

        """
        # padding index 0 is converted to ignore_index -1
        target = target - 1

        # (batch_size, length, codebook_size + 1) -> (batch_size, codebook_size + 1, length)
        input = input.transpose(-1, -2)

        return super().forward(input, target)
