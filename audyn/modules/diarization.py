import math
from itertools import combinations

import torch
import torch.nn as nn


class PowerSet(nn.Module):
    """Convert powerset classification scores to multi-label predictions.

    The input's class dimension contains scores for subsets of up to ``max_subset``
    classes, ordered first by subset size and then lexicographically. The output
    contains a binary activity value for each of the ``num_classes`` classes.
    """

    def __init__(self, num_classes: int, max_subset: int) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError("num_classes must be positive.")

        if not 0 <= max_subset <= num_classes:
            raise ValueError("max_subset must be between 0 and num_classes.")

        self.num_classes = num_classes
        self.max_subset = max_subset

        num_powerset_classes = 0

        for size in range(max_subset + 1):
            num_powerset_classes += math.comb(num_classes, size)

        self.num_powerset_classes = num_powerset_classes

        powerset_to_multilabel = torch.zeros(self.num_powerset_classes, num_classes)
        index = 0

        for size in range(max_subset + 1):
            for subset in combinations(range(num_classes), size):
                powerset_to_multilabel[index, list(subset)] = 1
                index += 1

        self.register_buffer("powerset_to_multilabel", powerset_to_multilabel, persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Select the highest-scoring subset for each frame.

        Args:
            input (torch.Tensor): Scores of shape
                ``(batch_size, num_powerset_classes, num_frames)``.

        Returns:
            torch.Tensor: Binary labels of shape
                ``(batch_size, num_classes, num_frames)``.

        """
        if input.dim() != 3:
            raise ValueError(f"Expected 3-dimensional input, but got {input.dim()} dimensions.")

        if input.size(1) != self.num_powerset_classes:
            raise ValueError(
                f"Expected {self.num_powerset_classes} powerset classes, but got {input.size(1)}."
            )

        indices = torch.argmax(input, dim=1)
        multilabel = self.powerset_to_multilabel[indices]
        output = multilabel.transpose(1, 2).contiguous()

        return output

    def extra_repr(self) -> str:
        return f"num_classes={self.num_classes}, max_subset={self.max_subset}"
