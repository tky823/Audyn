from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .activation import RelativePositionalMultiheadAttention
from .fastspeech import ConvBlock
from .fastspeech import FFTrBlock as BaseFFTrBlock
from .fastspeech import MultiheadSelfAttentionBlock as BaseMultiheadSelfAttentionBlock
from .glow import ActNorm1d, InvertiblePointwiseConv1d
from .waveglow import StackedResidualConvBlock1d, WaveNetAffineCoupling
from .wavenet import GatedConv1d
from .wavenet import ResidualConvBlock1d as BaseResidualConvBlock1d

__all__ = [
    "MaskedActNorm1d",
    "MaskedInvertiblePointwiseConv1d",
    "MaskedWaveNetAffineCoupling",
]


class GlowTTSFFTrBlock(BaseFFTrBlock):
    def __init__(
        self,
        d_model: int,
        hidden_channels: int,
        num_heads: int = 2,
        kernel_size: List[int] = [9, 1],
        dropout: float = 0.1,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        window_size: int = None,
        share_heads: bool = True,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super(BaseFFTrBlock, self).__init__()

        self.mha = MultiheadSelfAttentionBlock(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            window_size=window_size,
            share_heads=share_heads,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.ffn = ConvBlock(
            d_model, hidden_channels, kernel_size, dropout=dropout, **factory_kwargs
        )

        self.num_heads = num_heads
        self.batch_first = batch_first


class FFTrBlock(GlowTTSFFTrBlock):
    """Wrapper class of GlowTTSFFTrBlock."""


class MultiheadSelfAttentionBlock(BaseMultiheadSelfAttentionBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        window_size: int = None,
        share_heads: bool = True,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super(BaseMultiheadSelfAttentionBlock, self).__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.mha = RelativePositionalMultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            window_size=window_size,
            share_heads=share_heads,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.layer_norm = nn.LayerNorm(embed_dim, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        self.batch_first = batch_first


class MaskedActNorm1d(ActNorm1d):
    """ActNorm1d with padding mask.

    This module takes variable-length input.
    """

    @torch.no_grad()
    def _initialize_parameters(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> None:
        if padding_mask is None:
            super()._initialize_parameters(input)
        else:
            is_distributed = dist.is_available() and dist.is_initialized()

            # obtain world size for DDP
            if is_distributed:
                world_size = dist.get_world_size()
            else:
                world_size = 1

            expanded_padding_mask = _expand_padding_mask(padding_mask, input)
            expanded_non_padding_mask = torch.logical_not(expanded_padding_mask)
            num_elements = expanded_non_padding_mask.sum(dim=(0, 2))
            masked_input = input.masked_fill(expanded_padding_mask, 0)
            sum_input = torch.sum(masked_input, dim=(0, 2))

            if is_distributed:
                # gather statistics
                # sum_input:
                #  (num_features,) -> (num_gpus, num_features) -> (num_features,)
                # num_elements:
                #  (num_features,) -> (num_gpus, num_features) -> (num_features,)
                gathered_sum_input = [torch.zeros_like(sum_input) for _ in range(world_size)]
                gathered_num_elements = [torch.zeros_like(num_elements) for _ in range(world_size)]
                dist.all_gather(gathered_sum_input, sum_input)
                dist.all_gather(gathered_num_elements, num_elements)
                gathered_sum_input = torch.stack(gathered_sum_input, dim=0)
                gathered_num_elements = torch.stack(gathered_num_elements, dim=0)
                sum_input = torch.sum(gathered_sum_input, dim=0)
                num_elements = torch.sum(gathered_num_elements, dim=0)

            mean = sum_input / num_elements
            zero_mean_input = input - mean.unsqueeze(dim=-1)
            masked_zero_mean_input = torch.masked_fill(zero_mean_input, expanded_padding_mask, 0)
            sum_input = torch.sum(masked_zero_mean_input**2, dim=(0, 2))

            if is_distributed:
                # gather sum_input
                # (num_features,) -> (world_size, num_features)
                gathered_sum_input = [torch.zeros_like(sum_input) for _ in range(world_size)]
                dist.all_gather(gathered_sum_input, sum_input)
                gathered_sum_input = torch.stack(gathered_sum_input, dim=0)
                sum_input = torch.sum(gathered_sum_input, dim=0)

            log_std = 0.5 * (torch.log(sum_input) - torch.log(num_elements))
            bias = -mean * torch.exp(-log_std)

            self.log_scale.data.copy_(-log_std)
            self.bias.data.copy_(bias)

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
            log_scale = self.log_scale
            bias = self.bias
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)

            scale = torch.exp(log_scale)
            scale = scale.unsqueeze(dim=-1)
            bias = bias.unsqueeze(dim=-1)
            x = scale * input + bias
            output = x.masked_fill(expanded_padding_mask, 0)

            if logdet is not None:
                log_scale = log_scale.unsqueeze(dim=-1)
                log_scale = log_scale.expand(input.size())
                log_scale = log_scale.masked_fill(expanded_padding_mask, 0)
                logdet = logdet + log_scale.sum(dim=(1, 2))

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
            log_scale = self.log_scale
            bias = self.bias
            expanded_padding_mask = _expand_padding_mask(padding_mask, input)

            scale = torch.exp(-log_scale)
            scale = scale.unsqueeze(dim=-1)
            bias = bias.unsqueeze(dim=-1)
            x = scale * (input - bias)
            output = x.masked_fill(expanded_padding_mask, 0)

            if logdet is not None:
                log_scale = log_scale.unsqueeze(dim=-1)
                log_scale = log_scale.expand(input.size())
                log_scale = log_scale.masked_fill(expanded_padding_mask, 0)
                logdet = logdet - log_scale.sum(dim=(1, 2))

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
            logdet (torch.Tensor, optional): Log-determinant of shape (batch_size,).
            reverse (bool): If ``True``, reverse operation is applied. Default: ``False`

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

    def __init__(
        self,
        num_splits: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super(InvertiblePointwiseConv1d, self).__init__()

        assert num_splits % 2 == 0, "num_splits ({}) should be divisible by 2.".format(num_splits)

        self.num_splits = num_splits

        weight = torch.empty((num_splits, num_splits, 1), **factory_kwargs)
        self.weight = nn.Parameter(weight)

        self._reset_patameters()

    def extra_repr(self) -> str:
        s = "{num_splits}"

        return s.format(**self.__dict__)

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
        num_splits = self.num_splits
        weight = self.weight

        batch_size, num_features, num_frames = input.size()

        assert (
            num_features % num_splits == 0
        ), "num_features ({}) should be divisible by num_splits {}.".format(
            num_features, num_splits
        )

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if logdet is None:
            logabsdet = None
        else:
            _, logabsdet = torch.linalg.slogdet(weight.squeeze(dim=-1))

        x = input.view(batch_size * 2, num_features // num_splits, num_splits // 2, num_frames)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, num_splits, (num_features // num_splits) * num_frames)

        # num_elements: () or (batch_size,)
        if padding_mask is None:
            num_elements = num_frames
        else:
            if padding_mask.dim() != 2:
                raise ValueError(
                    f"Only 2D mask is supported, but {padding_mask.dim()}D mask is given."
                )

            non_padding_mask = torch.logical_not(padding_mask)
            num_elements = non_padding_mask.sum(dim=-1)

        if reverse:
            # use Gaussian elimination for numerical stability
            w = weight.squeeze(dim=-1)
            x = torch.linalg.solve(w, x)

            if logdet is not None:
                logdet = logdet - num_elements * logabsdet
        else:
            x = F.conv1d(x, weight, stride=1, dilation=1, groups=1)

            if logdet is not None:
                logdet = logdet + num_elements * logabsdet

        x = x.view(batch_size, 2, num_splits // 2, num_features // num_splits, num_frames)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        output = x.view(batch_size, num_features, num_frames)

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
        num_layers: int = 4,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        bias: bool = True,
        is_causal: bool = False,
        conv: str = "gated",
        dropout: float = 0,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        scaling_channels: Optional[int] = None,
    ) -> None:
        coupling = MaskedStackedResidualConvBlock1d(
            coupling_channels,
            hidden_channels,
            skip_channels=skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=1,
            dilation_rate=dilation_rate,
            bias=bias,
            is_causal=is_causal,
            conv=conv,
            dropout=dropout,
            weight_norm=weight_norm,
        )

        super(WaveNetAffineCoupling, self).__init__(
            coupling,
            split=split,
            concat=concat,
            scaling=scaling,
            scaling_channels=scaling_channels,
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of MaskedWaveNetAffineCoupling.

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
                scale = scale.unsqueeze(dim=-1)
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
                scale = scale.unsqueeze(dim=-1)
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
        num_layers: int = 4,
        kernel_size: int = 5,
        stride: int = 1,
        dilation_rate: int = 1,
        bias: bool = True,
        is_causal: bool = False,
        conv: str = "gated",
        dropout: float = 0,
        weight_norm: bool = True,
    ) -> None:
        # call nn.Module.__init__()
        super(StackedResidualConvBlock1d, self).__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.num_layers = num_layers

        self.bottleneck_conv1d_in = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
        )

        backbone = []

        for layer_idx in range(num_layers):
            if dilation_rate > 1:
                dilation = dilation_rate**layer_idx

                assert (
                    stride == 1
                ), "When dilated convolution, stride is expected to be 1, but {} is given.".format(
                    stride
                )
            else:
                dilation = 1

            if layer_idx < num_layers - 1:
                dual_head = True
            else:
                dual_head = False

            backbone.append(
                ResidualConvBlock1d(
                    hidden_channels,
                    hidden_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    bias=bias,
                    is_causal=is_causal,
                    dual_head=dual_head,
                    conv=conv,
                    dropout=dropout,
                    local_channels=None,
                    global_channels=None,
                    weight_norm=weight_norm,
                )
            )

        self.backbone = nn.ModuleList(backbone)
        self.bottleneck_conv1d_out = nn.Conv1d(
            skip_channels,
            2 * in_channels,
            kernel_size=1,
            stride=1,
        )

        # registered_weight_norms manages normalization status of backbone
        self.registered_weight_norms = set()

        if weight_norm:
            self.registered_weight_norms.add("backbone")
            self.weight_norm_()

        self._reset_parameters()

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


class ResidualConvBlock1d(BaseResidualConvBlock1d):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        is_causal: bool = True,
        dual_head: bool = True,
        conv: str = "gated",
        dropout: float = 0,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        # call nn.Module.__init__()
        super(BaseResidualConvBlock1d, self).__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.kernel_size, self.dilation = kernel_size, dilation
        self.is_causal = is_causal
        self.dual_head = dual_head

        if conv == "gated":
            assert stride == 1, f"stride is expected to 1, but given {stride}."

            self.conv1d = GatedConv1d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                bias=bias,
                is_causal=is_causal,
                local_channels=local_channels,
                global_channels=global_channels,
                weight_norm=weight_norm,
            )
        else:
            raise ValueError("{} is not supported for conv.".format(conv))

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if dual_head:
            self.output_conv1d = nn.Conv1d(
                hidden_channels, in_channels, kernel_size=1, stride=1, bias=bias
            )
        else:
            self.output_conv1d = None

        self.skip_conv1d = nn.Conv1d(
            hidden_channels, skip_channels, kernel_size=1, stride=1, bias=bias
        )

        # registered_weight_norms manages normalization status of conv1d
        self.registered_weight_norms = set()

        if weight_norm:
            self.registered_weight_norms.add("conv1d")
            self.weight_norm_()

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward pass of residual convolution block.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, num_frames).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, num_frames).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            Tuple of torch.Tensor containing:
            - torch.Tensor: Output of shape (batch_size, in_channels, num_frames).
            - torch.Tensor: Output of shape (batch_size, in_channels, num_frames).

        """
        residual = input

        x = self.conv1d(
            input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
        )

        if self.dropout is not None:
            x = self.dropout(x)

        skip = self.skip_conv1d(x)

        if self.dual_head:
            output = self.output_conv1d(x)
            output = residual + output
        else:
            output = None

        return output, skip


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
