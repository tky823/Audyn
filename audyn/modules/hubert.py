import math

import torch
import torch.nn as nn

__all__ = [
    "HuBERTGELU",
]


class HuBERTGELU(nn.Module):
    """GELU for HuBERT.

    .. note::

        GELU function by ``nn.GELU`` causes computational errors.
        It is recommended to use ``HuBERTGELU``.

    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        erf = torch.erf(input / math.sqrt(2.0))
        output = input * 0.5 * (1.0 + erf)

        return output
