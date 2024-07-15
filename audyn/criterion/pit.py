import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

__all__ = ["pit"]


def pit(
    criterion: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    input: torch.Tensor,
    target: torch.Tensor,
    permutations: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, torch.LongTensor]:
    """Wrapper function for permutation invariant training.

    Args:
        criterion (nn.Module or callable): Criterion to apply PIT.
        input (torch.Tensor): Input feature of shape (batch_size, num_sources, *).
        output (torch.Tensor): Target feature of shape (batch_size, num_sources, *).

    Returns:
        tuple: Tuple of tensors containing

            - torch.Tensor: Minimum loss for each data
            - torch.LongTensor: Permutation indices.

    .. note::

        ``criterion`` should return loss of shape (batch_size, num_sources).

    """
    assert (
        input.dim() >= 2
    ), "At least, 2D is required to dim of input, but {}D tensor is given.".format(input.dim())
    assert (
        target.dim() >= 2
    ), "At least, 2D is required to dim of target, but {}D tensor is given.".format(target.dim())

    if permutations is None:
        factory_kwargs = {
            "dtype": torch.long,
            "device": target.device,
        }
        num_sources = target.size(1)
        permutations = itertools.permutations(range(num_sources))
        permutations = list(permutations)
        permutations = torch.tensor(permutations, **factory_kwargs)

    num_permutations = len(permutations)
    possible_loss = []

    for idx in range(num_permutations):
        permutation = permutations[idx]
        loss = criterion(input, target[:, permutation])
        possible_loss.append(loss)

    possible_loss = torch.stack(possible_loss, dim=0)
    loss, indices = torch.min(possible_loss, dim=0)

    return loss, permutations[indices]


class PIT(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]],
        num_sources: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.criterion = criterion

        if num_sources is None:
            self.register_buffer("permutations", None)
        else:
            permutations = itertools.permutations(range(num_sources))
            permutations = torch.tensor(list(permutations), dtype=torch.long)
            self.register_buffer("permutations", permutations)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor
    ) -> Union[torch.Tensor, torch.LongTensor]:
        """Forward pass of PIT.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_sources, *).
            output (torch.Tensor): Target feature of shape (batch_size, num_sources, *).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Minimum loss for each data
                - torch.LongTensor: Permutation indices.

        """
        loss, permutation = pit(self.criterion, input, target, permutations=self.permutations)

        return loss, permutation
