import itertools

import torch
import torch.nn as nn

from ..functional.diarization import multilabel_to_powerset, powerset_to_multilabel

__all__ = [
    "PowersetPITLoss",
]


class PowersetPITLoss(nn.Module):
    """Compute a powerset classification loss after resolving speaker permutations.

    Args:
        num_classes (int): Number of multilabel speaker classes.
        max_subset (int): Maximum number of active speakers in a powerset class.

    """

    def __init__(self, num_classes: int, max_subset: int) -> None:
        super().__init__()

        if not isinstance(num_classes, int) or num_classes < 1:
            raise ValueError("num_classes must be a positive integer.")

        if not isinstance(max_subset, int) or not 0 <= max_subset <= num_classes:
            raise ValueError("max_subset must be an integer between 0 and num_classes.")

        permutations = itertools.permutations(range(num_classes))
        permutations = torch.tensor(list(permutations), dtype=torch.long)

        self.num_classes = num_classes
        self.max_subset = max_subset
        self.criterion = nn.CrossEntropyLoss()
        self.surrogate_criterion = nn.MSELoss(reduction="none")
        self.register_buffer("permutations", permutations, persistent=False)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss for powerset logits and binary multilabel targets.

        Args:
            input (torch.Tensor): Powerset logits of shape
                ``(batch_size, num_powerset_classes, num_frames)``.
            target (torch.Tensor): Binary activity of shape
                ``(batch_size, num_classes, num_frames)``.

        Returns:
            torch.Tensor: Scalar cross-entropy loss.

        """
        # TODO: Generalize beyond 3D frame-level powerset classification.
        if input.dim() != 3 or target.dim() != 3:
            raise ValueError("input and target must both be 3-dimensional tensors.")

        if target.size(1) != self.num_classes:
            raise ValueError(
                f"Expected {self.num_classes} target classes, but got {target.size(1)}."
            )

        if input.size(0) != target.size(0) or input.size(2) != target.size(2):
            raise ValueError("input and target must have matching batch and frame dimensions.")

        with torch.no_grad():
            multilabel_input = powerset_to_multilabel(
                input, num_classes=self.num_classes, max_subset=self.max_subset
            )
            multilabel_input = multilabel_input.to(dtype=input.dtype)
            multilabel_target = target.to(dtype=input.dtype)
            possible_surrogate_loss = []

            for permutation in self.permutations:
                surrogate_loss = self.surrogate_criterion(
                    multilabel_input, multilabel_target[:, permutation]
                )
                surrogate_loss = surrogate_loss.mean(dim=-1)
                possible_surrogate_loss.append(surrogate_loss.sum(dim=-1))

            possible_surrogate_loss = torch.stack(possible_surrogate_loss, dim=0)
            indices = torch.argmin(possible_surrogate_loss, dim=0)
            permutation = self.permutations[indices]
            permutation = permutation.unsqueeze(dim=-1).expand_as(target)
            permuted_target = torch.gather(target, dim=1, index=permutation)
            powerset_target = multilabel_to_powerset(
                permuted_target, num_classes=self.num_classes, max_subset=self.max_subset
            )

        return self.criterion(input, powerset_target)
