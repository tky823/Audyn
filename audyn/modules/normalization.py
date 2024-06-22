import numbers

import torch
import torch.nn as nn
from torch.nn.modules.normalization import _shape_t

from .glow import ActNorm1d

__all__ = [
    "ActNorm1d",
    "RMSNorm",
    "GlobalLayerNorm",
    "CumulativeLayerNorm1d",
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


class GlobalLayerNorm(nn.Module):
    """Global layer normalization.

    See "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation"
    https://arxiv.org/abs/1809.07454.
    """

    def __init__(self, num_features: int, eps: float = 1e-8) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.norm = nn.GroupNorm(1, num_features, eps=eps)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of GlobalLayerNorm.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Normalized feature of shape (batch_size, num_features, *).

        """
        output = self.norm(input)

        return output

    def __repr__(self) -> str:
        s = "{}".format(self.__class__.__name__)
        s += "({num_features}, eps={eps})"

        return s.format(**self.__dict__)


class CumulativeLayerNorm1d(nn.Module):
    """Cumulative layer normalization.

    See "Conv-TasNet: Surpassing ideal time-frequency magnitude masking for speech separation"
    https://arxiv.org/abs/1809.07454.
    """

    def __init__(self, num_features, eps: float = 1e-12) -> None:
        super().__init__()

        self.num_features = num_features
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(1, num_features, 1))
        self.bias = nn.Parameter(torch.Tensor(1, num_features, 1))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.weight.data.fill_(1)
        self.bias.data.fill_(0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of CumulativeLayerNorm1d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Normalized feature of shape (batch_size, num_features, *).

        """
        eps = self.eps

        batch_size, num_features, *length_shape = input.size()
        x = input.view(batch_size, num_features, -1)
        length = input.size(-1)

        step_sum = torch.sum(x, dim=1)
        step_squared_sum = torch.sum(x**2, dim=1)
        cum_sum = torch.cumsum(step_sum, dim=1)
        cum_squared_sum = torch.cumsum(step_squared_sum, dim=1)

        cum_num = torch.arange(
            num_features,
            num_features * (length + 1),
            num_features,
            device=input.device,
            dtype=torch.float,
        )
        cum_mean = cum_sum / cum_num
        cum_squared_mean = cum_squared_sum / cum_num
        cum_var = cum_squared_mean - cum_mean**2

        cum_mean = cum_mean.unsqueeze(dim=1)
        cum_var = cum_var.unsqueeze(dim=1)

        x = (x - cum_mean) / (torch.sqrt(cum_var) + eps) * self.weight + self.bias
        output = x.view(batch_size, num_features, *length_shape)

        return output

    def __repr__(self) -> str:
        s = "{}".format(self.__class__.__name__)
        s += "({num_features}, eps={eps})"

        return s.format(**self.__dict__)
