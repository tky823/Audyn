import math
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import _IncompatibleKeys

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
    """ActNorm proposed in Glow.

    Args:
        num_features (int): Number of features.

    """

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

        self.log_scale = nn.Parameter(torch.empty((num_features,), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty((num_features,), **factory_kwargs))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.log_scale.data)
        nn.init.zeros_(self.bias.data)
        self.is_initialized = False

    @torch.no_grad()
    def _initialize_parameters(self, input: torch.Tensor) -> None:
        is_distributed = dist.is_available() and dist.is_initialized()

        # obtain world size for DDP
        if is_distributed:
            world_size = dist.get_world_size()
        else:
            world_size = 1

        batch_size, _, length = input.size()
        sum_input = torch.sum(input, dim=(0, 2))

        if is_distributed:
            # gather sum_input
            # (num_features,) -> (world_size, num_features)
            gathered_sum_input = [torch.zeros_like(sum_input) for _ in range(world_size)]
            dist.all_gather(gathered_sum_input, sum_input)
            gathered_sum_input = torch.stack(gathered_sum_input, dim=0)
            sum_input = torch.sum(gathered_sum_input, dim=0)

        mean = sum_input / (world_size * batch_size * length)
        zero_mean_input = input - mean.unsqueeze(dim=-1)
        sum_input = torch.sum(zero_mean_input**2, dim=(0, 2))

        if is_distributed:
            # gather sum_input
            # (num_features,) -> (world_size, num_features)
            gathered_sum_input = [torch.zeros_like(sum_input) for _ in range(world_size)]
            dist.all_gather(gathered_sum_input, sum_input)
            gathered_sum_input = torch.stack(gathered_sum_input, dim=0)
            sum_input = torch.sum(gathered_sum_input, dim=0)

        log_std = 0.5 * (torch.log(sum_input) - math.log((world_size * batch_size * length)))
        bias = -mean * torch.exp(-log_std)

        self.log_scale.data.copy_(-log_std)
        self.bias.data.copy_(bias)

        self.is_initialized = True

    def _forward(
        self,
        input: torch.Tensor,
        logdet: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_scale = self.log_scale
        bias = self.bias
        length = input.size(-1)

        scale = torch.exp(log_scale)
        scale = scale.unsqueeze(dim=-1)
        bias = bias.unsqueeze(dim=-1)
        output = scale * input + bias

        if logdet is not None:
            logdet = logdet + length * log_scale.sum()

        return output, logdet

    def _reverse(
        self,
        input: torch.Tensor,
        logdet: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_scale = self.log_scale
        bias = self.bias
        length = input.size(-1)

        scale = torch.exp(-log_scale)
        scale = scale.unsqueeze(dim=-1)
        bias = bias.unsqueeze(dim=-1)
        output = scale * (input - bias)

        if logdet is not None:
            logdet = logdet - length * log_scale.sum()

        return output, logdet

    def forward(
        self,
        input: torch.Tensor,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of ActNorm1d.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, num_features, length).
            logdet (torch.Tensor, optional): Log-determinant of shape (batch_size,).
            reverse (bool): If ``True``, reverse operation is applied. Default: ``False`

        Returns:
            torch.Tensor: Transformed tensor of same shape as input.

        """
        if self.training and not self.is_initialized:
            self._initialize_parameters(input)

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if reverse:
            output, logdet = self._reverse(input, logdet=logdet)
        else:
            output, logdet = self._forward(input, logdet=logdet)

        if logdet is None:
            return output
        else:
            return output, logdet

    def state_dict(
        self,
        destination: Dict[str, Any] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> Dict[str, Any]:
        """Return state_dict of module.

        .. note::

            Returned ``state_dict`` includes ``is_initialized`` flag.
            In terms of simplicity, registering ``is_initialized`` as boolean tensor
            is better, but it is incompatible with DDP.

        """
        state_dict = super().state_dict(
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        state_dict.update({prefix + "is_initialized": self.is_initialized})

        return state_dict

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True
    ) -> _IncompatibleKeys:
        is_initialized_key = "is_initialized"

        if is_initialized_key in state_dict.keys():
            is_initialized = state_dict.pop(is_initialized_key)

            if isinstance(is_initialized, torch.Tensor):
                # for backward compatibility
                is_initialized = is_initialized.item()

            self.is_initialized = is_initialized

        return super().load_state_dict(state_dict, strict=strict)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> Any:
        is_initialized_key = prefix + "is_initialized"

        if is_initialized_key in state_dict.keys():
            is_initialized = state_dict.pop(is_initialized_key)

            if isinstance(is_initialized, torch.Tensor):
                # for backward compatibility
                is_initialized = is_initialized.item()

            self.is_initialized = is_initialized

        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
