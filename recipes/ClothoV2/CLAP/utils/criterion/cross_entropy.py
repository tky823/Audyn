from typing import Optional

import torch
import torch.nn as nn


class MaskedLaguageModelCrossEntropyLoss(nn.CrossEntropyLoss):
    """Cross entropy loss for masked laguage model.

    Args:
        mask_index (int): Index of mask token. This is used to shift indices of tokens.
        batch_first (int): If ``True``, ``input`` is treated as
            (batch_size, max_length, num_classes). Otherwise, it is treated as
            (max_length, batch_size, num_classes).

    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        mask_index: int = 0,
        ignore_index: int = -100,
        reduction: str = "mean",
        batch_first: bool = True,
    ) -> None:
        super().__init__(weight, ignore_index=ignore_index, reduction=reduction)

        self.mask_index = mask_index
        self.batch_first = batch_first

        assert self.mask_index >= 0, "Mask index should be non-negative."
        assert self.ignore_index < 0, "Ignore index should be negative."

    def forward(
        self,
        input: torch.Tensor,
        target: torch.LongTensor,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of cross entropy loss for masked laguage model.

        Args:
            mask_index (int): Index of mask token. This is used to shift indices of tokens.
            input (torch.Tensor): Sequence of shape (batch_size, max_length, num_classes)
                or (max_length, batch_size, num_classes).
            target (torch.LongTensor): Target indices of shape (batch_size, max_length)
                or (max_length, batch_size). Indices larger than ``mask_index`` is shited
                to be compatible with num_classes.

        """
        mask_index = self.mask_index
        batch_first = self.batch_first

        factory_kwargs = {
            "device": target.device,
            "dtype": torch.long,
        }

        if batch_first:
            batch_size, max_length, _ = input.size()
            input = input.permute(0, 2, 1)
        else:
            max_length, batch_size, _ = input.size()
            input = input.permute(1, 2, 0)

        target = torch.where(target >= mask_index, target - 1, target)

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)
        target = target.masked_fill(padding_mask, self.ignore_index)

        return super().forward(input, target)
