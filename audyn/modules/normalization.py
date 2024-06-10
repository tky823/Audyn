import numbers

import torch
import torch.nn as nn
from torch.nn.modules.normalization import _shape_t

from .glow import ActNorm1d

__all__ = [
    "ActNorm1d",
    "RMSNorm",
]


class RMSNorm(nn.Module):
    """Root mean square layer normalization.

    See https://arxiv.org/abs/1910.07467 for details.
    """

    # This implementation based on nn.LayerNorm.

    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            weight = torch.empty(self.normalized_shape, **factory_kwargs)
            self.weight = nn.Parameter(weight)

            if bias:
                bias = torch.empty(self.normalized_shape, **factory_kwargs)
                self.bias = nn.Parameter(bias)
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        normalized_shape = self.normalized_shape
        eps = self.eps

        dim = tuple(range(-1, -len(normalized_shape) - 1, -1))
        squared_mean = torch.mean(input**2, dim=dim, keepdim=True)
        x = input / torch.sqrt(squared_mean + eps)

        if self.bias is None:
            output = self.weight * x
        else:
            output = self.weight * x + self.bias

        return output
