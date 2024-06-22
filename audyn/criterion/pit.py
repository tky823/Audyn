import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn as nn

__all__ = ["pit"]


def pit(
    criterion: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    input: torch.Tensor,
    target: torch.Tensor,
    patterns: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, torch.Tensor]:
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
    if patterns is None:
        factory_kwargs = {
            "dtype": torch.long,
            "device": target.device,
        }
        num_sources = target.size(1)
        patterns = itertools.permutations(range(num_sources))
        patterns = list(patterns)
        patterns = torch.tensor(patterns, **factory_kwargs)

    num_patterns = len(patterns)
    possible_loss = []

    for idx in range(num_patterns):
        pattern = patterns[idx]
        loss = criterion(input, target[:, pattern])
        possible_loss.append(loss)

    possible_loss = torch.stack(possible_loss, dim=0)
    loss, indices = torch.min(possible_loss, dim=0)

    return loss, patterns[indices]


class PIT(nn.Module):
    def __init__(
        self,
        criterion: Union[nn.Module, Callable[[torch.Tensor, torch.Tensor]]],
        num_sources: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.criterion = criterion

        if num_sources is None:
            self.register_buffer("patterns", None)
        else:
            patterns = itertools.permutations(range(num_sources))
            patterns = torch.tensor(list(patterns), dtype=torch.long)
            self.register_buffer("patterns", patterns)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of PIT.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_sources, *).
            output (torch.Tensor): Target feature of shape (batch_size, num_sources, *).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Minimum loss for each data
                - torch.LongTensor: Permutation indices.

        """
        loss, pattern = pit(self.criterion, input, target, patterns=self.patterns)

        return loss, pattern
