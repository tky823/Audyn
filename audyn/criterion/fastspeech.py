from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

__all__ = ["FastSpeechMSELoss"]


class FastSpeechMSELoss(nn.Module):
    def __init__(
        self,
        take_log: Union[bool, Dict[str, bool]] = False,
        reduction: Optional[str] = None,
        batch_first: bool = False,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        super().__init__()

        if isinstance(take_log, bool):
            self.take_log_input = take_log
            self.take_log_target = take_log
        elif isinstance(take_log, dict) or isinstance(take_log, DictConfig):
            self.take_log_input = take_log["input"]
            self.take_log_target = take_log["target"]
        else:
            raise ValueError(f"Invalid type {type(take_log)} is given to take_log.")

        if reduction is None:
            reduction = "none"

        self.reduction = reduction
        self.batch_first = batch_first

        self.kwargs_clamp = {}

        if min is None:
            self.kwargs_clamp["min"] = 0
        else:
            self.kwargs_clamp["min"] = min

        if max is not None:
            self.kwargs_clamp["max"] = max

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        length: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of MSE loss used in paper of FastSpeech.

        Args:
            input (torch.Tensor): Estimated feature of shape (batch_size, length, \*)
                if ``batch_first=True``, otherwise (length, batch_size, \*).
            target (torch.Tensor): Target feature of shape (batch_size, length, \*)
                if ``batch_first=True``, otherwise (length, batch_size, \*).
            length (torch.BoolTensor, optional): Padding mask of shape (batch_size,)
                or (length,).

        Returns:
            torch.Tensor: Mean squared error. If ``reduction=None``, shape is same as input.
                If ``reduction=mean``, shape is ().

        """
        reduction = self.reduction

        if self.take_log_input:
            input = torch.clamp(input.float(), **self.kwargs_clamp)
            input = torch.log(input)

        if self.take_log_target:
            target = torch.clamp(target.float(), **self.kwargs_clamp)
            target = torch.log(target)

        loss = (input - target) ** 2

        if length is None:
            if reduction == "mean":
                loss = loss.mean()
            elif reduction == "sum":
                loss = loss.sum()
            elif reduction != "none":
                raise ValueError(f"reduction={reduction} is not supported.")
        else:
            n_dims = loss.dim()
            newdims = (n_dims - 2) * (1,)

            if self.batch_first:
                batch_size, max_length = loss.size()[:2]
                padding_mask = self._make_padding_mask(length, max_length=max_length)
                padding_mask = padding_mask.view(batch_size, max_length, *newdims)
            else:
                max_length, batch_size = loss.size()[:2]
                padding_mask = self._make_padding_mask(length, max_length=max_length)
                padding_mask = padding_mask.permute(1, 0)
                padding_mask = padding_mask.view(max_length, batch_size, *newdims)

            loss = loss.masked_fill(padding_mask, 0)

            if reduction == "mean":
                non_padding_mask = torch.logical_not(padding_mask)
                non_padding_mask = non_padding_mask.expand(loss.size())
                loss = loss.sum() / non_padding_mask.sum()
            elif reduction == "sum":
                loss = loss.sum()
            elif reduction != "none":
                raise ValueError(f"reduction={reduction} is not supported.")

        return loss

    @staticmethod
    def _make_padding_mask(
        length: torch.LongTensor, max_length: Optional[int] = None
    ) -> torch.BoolTensor:
        if max_length is None:
            max_length = torch.max(length)

        indices = torch.arange(max_length).to(length.device)
        padding_mask = indices >= length.unsqueeze(dim=-1)

        return padding_mask
