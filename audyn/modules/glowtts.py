from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .glow import ActNorm1d, InvertiblePointwiseConv1d
from .waveglow import StackedResidualConvBlock1d, WaveNetAffineCoupling

__all__ = [
    "MaskedActNorm1d",
    "MaskedInvertiblePointwiseConv1d",
    "MaskedWaveNetAffineCoupling",
]


class MaskedActNorm1d(ActNorm1d):
    """ActNorm1d with padding mask.

    This module takes variable-length input.
    """

    def _initialize_parameters(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        if padding_mask is None:
            super()._initialize_parameters(input)
        else:
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)
            expanded_non_padding_mask = torch.logical_not(expanded_padding_mask)
            num_elements = expanded_non_padding_mask.sum(dim=(0, 2))
            masked_input = input.masked_fill(expanded_padding_mask, 0)
            mean = masked_input.sum(dim=(0, 2)) / num_elements

            zero_mean_input = masked_input - mean.unsqueeze(dim=-1)
            squared_input = torch.masked_fill(zero_mean_input**2, expanded_padding_mask, 0)
            log_std = 0.5 * (torch.log(squared_input.sum(dim=(0, 2))) - torch.log(num_elements))

            self.log_std.data.copy_(log_std)
            self.mean.data.copy_(mean)

            self.is_initialized = True

    def _forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            output, logdet = super()._forward(input, logdet=logdet)
        else:
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)
            expanded_non_padding_mask = torch.logical_not(expanded_padding_mask)
            # count elements per batch dimension
            num_elements = expanded_non_padding_mask.sum(dim=(1, 2))

            log_std = self.log_std.unsqueeze(dim=-1)
            std = torch.exp(log_std)
            mean = self.mean.unsqueeze(dim=-1)
            x = (input - mean) / std
            output = x.masked_fill(expanded_padding_mask, 0)

            if logdet is not None:
                log_std = log_std.expand(input.size())
                log_std = log_std.masked_fill(expanded_padding_mask, 0)
                logdet = logdet - num_elements * log_std.sum(dim=(1, 2))

        return output, logdet

    def _reverse(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if padding_mask is None:
            output, logdet = super()._reverse(input, logdet=logdet)
        else:
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)
            expanded_non_padding_mask = torch.logical_not(expanded_padding_mask)
            # count elements per batch dimension
            num_elements = expanded_non_padding_mask.sum(dim=(1, 2))

            log_std = self.log_std.unsqueeze(dim=-1)
            std = torch.exp(log_std)
            mean = self.mean.unsqueeze(dim=-1)
            x = std * input + mean
            output = x.masked_fill(expanded_padding_mask, 0)

            if logdet is not None:
                log_std = log_std.expand(input.size())
                log_std = log_std.masked_fill(expanded_padding_mask, 0)
                logdet = logdet + num_elements * log_std.sum(dim=(1, 2))

        return output, logdet

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of MaskedActNorm1d.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, num_features, length).
            padding_mask (torch.BoolTensor): Padding mask of shape
                (batch_size, length) or (batch_size, num_features, length).

        Returns:
            torch.Tensor: Transformed tensor of same shape as input.

        """
        if self.training and not self.is_initialized:
            self._initialize_parameters(input, padding_mask=padding_mask)

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if reverse:
            output, logdet = self._reverse(
                input,
                padding_mask=padding_mask,
                logdet=logdet,
            )
        else:
            output, logdet = self._forward(
                input,
                padding_mask=padding_mask,
                logdet=logdet,
            )

        if logdet is None:
            return output
        else:
            return output, logdet


class MaskedInvertiblePointwiseConv1d(InvertiblePointwiseConv1d):
    """InvertiblePointwiseConv1d with padding mask.

    This module takes variable-length input.
    """

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of MaskedInvertiblePointwiseConv1d.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, num_features, length).
            padding_mask (torch.BoolTensor): Padding mask of shape (batch_size, length).
                3D mask is not supported.

        Returns:
            torch.Tensor: Transformed tensor of same shape as input.

        .. note::

            To handle 3D padding mask properly, we need to refine formulation.

        """
        weight = self.weight

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if logdet is None:
            logabsdet = None
        else:
            _, logabsdet = torch.linalg.slogdet(weight.squeeze(dim=-1))

        # num_frames: () or (batch_size,)
        if padding_mask is None:
            num_frames = input.size(-1)
        else:
            if padding_mask.dim() != 2:
                raise ValueError(
                    f"Only 2D mask is supported, but {padding_mask.dim()}D mask is given."
                )

            non_padding_mask = torch.logical_not(padding_mask)
            num_frames = non_padding_mask.sum(dim=-1)

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


class MaskedWaveNetAffineCoupling(WaveNetAffineCoupling):
    """WaveNetAffineCoupling with padding mask.

    This module takes variable-length input.
    """

    def __init__(
        self,
        coupling_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 3,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        in_channels: Optional[int] = None,
    ) -> None:
        coupling = MaskedStackedResidualConvBlock1d(
            coupling_channels,
            hidden_channels,
            skip_channels=skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=1,
            dilated=True,
            bias=bias,
            causal=causal,
            conv=conv,
            weight_norm=weight_norm,
        )

        super(WaveNetAffineCoupling, self).__init__(
            coupling,
            split=split,
            concat=concat,
            scaling=scaling,
            in_channels=in_channels,
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of MaskedInvertiblePointwiseConv1d.

        Args:
            input (torch.Tensor): Tensor of shape (batch_size, num_features, length).
            padding_mask (torch.BoolTensor): Padding mask of shape (batch_size, length).
                3D mask is not supported.

        Returns:
            torch.Tensor: Transformed tensor of same shape as input.

        .. note::

            To handle 3D padding mask properly, we need to refine formulation.

        """
        assert input.dim() == 3, "input is expected to be 3D tensor, but given {}D tensor.".format(
            input.dim()
        )
        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if padding_mask is not None and padding_mask.dim() != 2:
            raise ValueError(
                f"Only 2D mask is supported, but {padding_mask.dim()}D mask is given."
            )

        if reverse:
            if padding_mask is not None:
                expanded_padding_mask = _expand_padding_mask(padding_mask, input)
                y = input.masked_fill(expanded_padding_mask, 0)
            else:
                y = input

            y1, y2 = self.call_split(y)
            log_s, t = self.coupling(y1, padding_mask=padding_mask)

            if padding_mask is not None:
                expanded_padding_mask = _expand_padding_mask(padding_mask, log_s)
                log_s = log_s.masked_fill(expanded_padding_mask, 0)

                expanded_padding_mask = _expand_padding_mask(padding_mask, t)
                t = t.masked_fill(expanded_padding_mask, 0)

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            x1 = y1
            x2 = (y2 - t) * torch.exp(-log_s)
            output = self.call_concat(x1, x2)

            if logdet is not None:
                # Padding mask is applied to log_s in self.coupling.
                logdet = logdet - log_s.sum(dim=(1, 2))
        else:
            if padding_mask is not None:
                expanded_padding_mask = _expand_padding_mask(padding_mask, input)
                x = input.masked_fill(expanded_padding_mask, 0)
            else:
                x = input

            x1, x2 = self.call_split(x)
            log_s, t = self.coupling(x1, padding_mask=padding_mask)

            if padding_mask is not None:
                expanded_padding_mask = _expand_padding_mask(padding_mask, log_s)
                log_s = log_s.masked_fill(expanded_padding_mask, 0)

                expanded_padding_mask = _expand_padding_mask(padding_mask, t)
                t = t.masked_fill(expanded_padding_mask, 0)

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            y1 = x1
            y2 = x2 * torch.exp(log_s) + t
            output = self.call_concat(y1, y2)

            if logdet is not None:
                # Padding mask is applied to log_s in self.coupling.
                logdet = logdet + log_s.sum(dim=(1, 2))

        if logdet is None:
            return output
        else:
            return output, logdet


class MaskedStackedResidualConvBlock1d(StackedResidualConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 3,
        stride: int = 1,
        dilated: bool = True,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        weight_norm: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            hidden_channels,
            skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=stride,
            dilated=dilated,
            bias=bias,
            causal=causal,
            conv=conv,
            local_dim=None,
            global_dim=None,
            weight_norm=weight_norm,
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_channels = self.in_channels

        if padding_mask is not None:
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)
            x = input.masked_fill(expanded_padding_mask, 0)
        else:
            x = input

        x = self.bottleneck_conv1d_in(x)
        skip_connection = 0

        for layer_idx in range(self.num_layers):
            x, skip = self.backbone[layer_idx](
                x,
                local_conditioning=None,
                global_conditioning=None,
            )
            if padding_mask is not None and x is not None:
                expanded_padding_mask = _expand_padding_mask(padding_mask, x)
                x = x.masked_fill(expanded_padding_mask, 0)

            skip_connection = skip_connection + skip

        output = self.bottleneck_conv1d_out(skip_connection)

        if padding_mask is not None:
            expanded_padding_mask = _expand_padding_mask(padding_mask, output)
            output = output.masked_fill(expanded_padding_mask, 0)

        log_s, t = torch.split(output, [in_channels, in_channels], dim=1)

        return log_s, t


def _expand_padding_mask(
    padding_mask: torch.BoolTensor,
    other: torch.Tensor,
) -> torch.BoolTensor:
    """Expand padding mask.

    Args:
        padding_mask (torch.BoolTensor): Padding mask of shape
            (batch_size, length) or (batch_size, num_features, length).
        other (torch.Tensor): Tensor of shape (batch_size, num_features, length).

    Returns:
        torch.BoolTensor: Expanded padding mask of shape (batch_size, num_features, length).

    """
    if padding_mask.dim() == 2:
        padding_mask = padding_mask.unsqueeze(dim=1)
    elif padding_mask.dim() != 3:
        raise ValueError(f"{padding_mask.dim()}D mask is not supported.")

    expanded_padding_mask = padding_mask.expand(other.size())

    return expanded_padding_mask
