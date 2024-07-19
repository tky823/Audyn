import copy
import math
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LoRALinear",
]


class LoRALinear(nn.Module):
    """Linear layer for low-rank adaptation.

    Args:
        weight (nn.Parameter or torch.Tensor, optional): Weight parameter in ``nn.Linear``.
        bias (nn.Parameter or torch.Tensor, optional): Bias parameter in ``nn.Linear``.
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

    """

    def __init__(
        self,
        weight: Union[nn.Parameter, torch.Tensor],
        bias: Optional[Union[nn.Parameter, torch.Tensor]] = None,
        rank: int = 8,
        persistent: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

        super().__init__()

        weight = copy.copy(weight.data)

        if bias is not None:
            bias = copy.copy(bias.data)

        # register weight and bias as buffer
        self.register_buffer("weight", weight, persistent=persistent)
        self.register_buffer("bias", bias, persistent=persistent)

        out_features, in_features = weight.size()

        self.weight_in = nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))
        self.weight_out = nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.linear(input, self.weight, bias=self.bias)
        x_lora = F.linear(input, self.weight_in)
        x_lora = F.linear(x_lora, self.weight_out)
        output = x + x_lora

        return output

    def _reset_parameters(self) -> None:
        std = 1 / math.sqrt(self.rank)
        self.weight_in.data.normal_(std=std)
        self.weight_out.data.zero_()

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", bias={self.bias is not None}"
        s += f", rank={self.rank}"

        return s
