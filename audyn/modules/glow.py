from collections import OrderedDict
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow import BaseFlow

__all__ = [
    "InvertiblePointwiseConv1d",
    "InvertiblePointwiseConv2d",
    "ActNorm1d",
]


class InvertiblePointwiseConv1d(BaseFlow):
    def __init__(
        self,
        num_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features

        weight = torch.empty((num_features, num_features, 1), **factory_kwargs)
        self.weight = nn.Parameter(weight)

        self._reset_patameters()

    def _reset_patameters(self):
        nn.init.orthogonal_(self.weight)

    def extra_repr(self) -> str:
        s = "{num_features}"

        return s.format(**self.__dict__)

    def forward(
        self, input: torch.Tensor, logdet: torch.Tensor = None, reverse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of InvertiblePointwiseConv1d.

        Args:
            input (torch.Tensor): 3D tensor of shape (batch_size, num_features, length).
            logdet (torch.Tensor, optional): Lod-determinant of shape (batch_size,).
            reverse (bool): If ``True``, reverse process is computed.

        Returns:
            torch.Tensor: Transformed output having shape as input.

        """
        weight = self.weight
        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if logdet is not None:
            num_frames = input.size(-1)
            _, logabsdet = torch.linalg.slogdet(weight.squeeze(dim=-1))

        if reverse:
            # use Gaussian elimination for numerical stability
            w = weight.squeeze(dim=-1)
            output = torch.linalg.solve(w, input)

            if logdet is not None:
                logdet = logdet - num_frames * logabsdet
        else:
            output = F.conv1d(input, weight, stride=1, dilation=1, groups=1)

            if logdet is not None:
                logdet = logdet + num_frames * logabsdet

        if logdet is None:
            return output
        else:
            return output, logdet


class InvertiblePointwiseConv2d(BaseFlow):
    def __init__(
        self,
        num_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_features = num_features

        weight = torch.empty((num_features, num_features, 1, 1), **factory_kwargs)
        self.weight = nn.Parameter(weight)

        self._reset_patameters()

    def _reset_patameters(self):
        nn.init.orthogonal_(self.weight)

    def extra_repr(self) -> str:
        s = "{num_features}"

        return s.format(**self.__dict__)

    def forward(
        self, input: torch.Tensor, logdet: torch.Tensor = None, reverse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of InvertiblePointwiseConv2d.

        Args:
            input (torch.Tensor): 4D tensor of shape (batch_size, num_features, height, width).
            logdet (torch.Tensor, optional): Lod-determinant of shape (batch_size,).
            reverse (bool): If ``True``, reverse process is computed.

        Returns:
            torch.Tensor: Transformed output having shape as input.

        """
        weight = self.weight
        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if logdet is not None:
            height, width = input.size()[-2:]
            _, logabsdet = torch.linalg.slogdet(weight.squeeze(dim=-1).squeeze(dim=-1))

        if reverse:
            # use Gaussian elimination for numerical stability
            batch_size, num_features, height, width = input.size()
            x = input.view(batch_size, num_features, height * width)
            w = weight.view(num_features, num_features)
            x = torch.linalg.solve(w, x)
            output = x.view(batch_size, num_features, height, width)

            if logdet is not None:
                logdet = logdet - height * width * logabsdet
        else:
            output = F.conv2d(input, weight, stride=1, dilation=1, groups=1)

            if logdet is not None:
                logdet = logdet + height * width * logabsdet

        if logdet is None:
            return output
        else:
            return output, logdet


class ActNorm1d(BaseFlow):
    def __init__(
        self,
        num_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.log_std = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
        self.mean = nn.Parameter(torch.empty((num_features,), **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.log_std.data)
        nn.init.zeros_(self.mean.data)
        self.is_initialized = False

    def _initialize_parameters(self, input: torch.Tensor) -> None:
        std, mean = torch.std_mean(input, dim=(0, 2), unbiased=False)
        mean = mean.detach()
        log_std = torch.log(std.detach())

        self.log_std.data.copy_(log_std)
        self.mean.data.copy_(mean)

        self.is_initialized = True

    def forward(
        self,
        input: torch.Tensor,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        length = input.size(-1)

        if self.training and not self.is_initialized:
            self._initialize_parameters(input)

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        std = torch.exp(self.log_std)
        std = std.unsqueeze(dim=-1)
        mean = self.mean.unsqueeze(dim=-1)

        if reverse:
            output = input * std + mean
        else:
            output = (input - mean) / std

        if logdet is not None:
            if reverse:
                logdet = logdet + length * self.log_std.sum()
            else:
                logdet = logdet - length * self.log_std.sum()

        if logdet is None:
            return output
        else:
            return output, logdet

    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        state_dict: OrderedDict = super().state_dict()
        state_dict["is_initialized"] = self.is_initialized

        return state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        is_initialized = state_dict["is_initialized"]
        self.is_initialized = is_initialized

        del state_dict["is_initialized"]

        return super().load_state_dict(state_dict, strict)
