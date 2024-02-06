from typing import Optional

import torch
import torch.nn as nn


class MaskedLaguageModelCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        batch_first: bool = True,
    ) -> None:
        super().__init__(weight, ignore_index=ignore_index, reduction=reduction)

        self.batch_first = batch_first

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, length: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_first = self.batch_first

        factory_kwargs = {
            "device": target.device,
            "dtype": torch.long,
        }

        if batch_first:
            batch_size, max_length, _ = input.size()
            max_length = max_length - 1
            _, x = torch.split(input, [1, max_length], dim=1)
            x = x.permute(0, 2, 1)
        else:
            max_length, batch_size, _ = input.size()
            max_length = max_length - 1
            _, x = torch.split(input, [1, max_length], dim=0)
            x = x.permute(1, 2, 0)

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)
        target = target.masked_fill(padding_mask, self.ignore_index)

        return super().forward(x, target)
