from typing import Any, List, Union

import torch
import torch.nn as nn

from .gan import DiscriminatorHingeLoss as BaseDiscriminatorHingeLoss
from .lsgan import MSELoss as BaseMSELoss

__all__ = ["FeatureMatchingLoss", "DiscriminatorHingeLoss"]


class MSELoss(BaseMSELoss):
    """Extension of MSELoss used for HiFi-GAN.

    This criterion receives both list objects.
    """

    def __init__(self, target: float, reduction: str = "mean") -> None:
        super().__init__(target, reduction=reduction)

    def forward(self, input: Union[torch.Tensor, List[Any]]) -> torch.Tensor:
        loss = 0

        if type(input) is list:
            for idx in range(len(input)):
                loss = loss + super().forward(input[idx])
        elif isinstance(input, torch.Tensor):
            loss = loss + super().forward(input)
        else:
            raise TypeError("Invalid type {} is found.".format(type(input)))

        return loss


class DiscriminatorHingeLoss(BaseDiscriminatorHingeLoss):
    def __init__(self, minimize: bool, margin: float = 1, reduction: str = "mean") -> None:
        super().__init__(minimize, margin, reduction)

    def forward(self, input: Union[torch.Tensor, List[Any]]) -> torch.Tensor:
        loss = 0

        if type(input) is list:
            for idx in range(len(input)):
                loss = loss + super().forward(input[idx])
        elif isinstance(input, torch.Tensor):
            loss = loss + super().forward(input)
        else:
            raise TypeError("Invalid type {} is found.".format(type(input)))

        return loss


class FeatureMatchingLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, input: List[List[torch.Tensor]], target: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        loss = self._forward(input, target)

        return loss

    def _forward(
        self, input: Union[torch.Tensor, List], target: Union[torch.Tensor, List]
    ) -> torch.Tensor:
        loss = 0

        if isinstance(input, torch.Tensor) or isinstance(target, torch.Tensor):
            loss = torch.abs(input - target)
            loss = loss.mean()
        elif isinstance(input, list) or isinstance(target, list):
            for _input, _target in zip(input, target):
                loss = loss + self._forward(_input, _target)
        else:
            raise TypeError("Invalid type ({} and {}) is given.".format(type(input), type(target)))

        return loss
