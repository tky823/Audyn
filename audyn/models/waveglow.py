import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.flow import AffineCoupling, BaseFlow
from ..modules.glow import InvertiblePointwiseConv1d
from .wavenet import ResidualConvBlock1d

__all__ = ["WaveGlow", "MultiSpeakerWaveGlow"]


class WaveGlow(BaseFlow):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_flows: int = 3,
        num_stacks: int = 4,
        num_layers: int = 8,
        num_groups: int = 8,
        early_size: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        upsample: Optional[nn.Module] = None,
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.num_flows = num_flows
        self.num_groups = num_groups
        self.early_size = early_size
        self.local_dim = local_dim

        if local_dim is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."
            local_dim = local_dim * num_groups

        self.upsample = upsample

        backbone = []

        for flow_idx in range(num_flows):
            num_features = in_channels * num_groups - flow_idx * early_size

            backbone.append(
                StackedWaveGlowBlock(
                    num_features,
                    hidden_channels,
                    skip_channels=skip_channels,
                    num_stacks=num_stacks,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    conv=conv,
                    local_dim=local_dim,
                    global_dim=global_dim,
                    weight_norm=weight_norm,
                )
            )

        self.backbone = nn.ModuleList(backbone)

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Forward pass of WaveGlow.

        Args:
            input (torch.Tensor): 3D tensor of shape (batch_size, in_channels, length).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_dim) or (batch_size, global_dim, 1).

        Returns:
            tuple of torch.Tensor.

        """
        num_groups = self.num_groups
        local_dim = self.local_dim

        batch_size, in_channels, length = input.size()
        padding = (num_groups - length % num_groups) % num_groups
        x = F.pad(input, (0, padding))
        x = x.view(batch_size, in_channels, (length + padding) // num_groups, num_groups)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(batch_size, in_channels * num_groups, (length + padding) // num_groups)

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)
        return_logdet = logdet is not None

        if local_conditioning is None:
            h_local = None
        else:
            h_local = self.upsample(local_conditioning)
            local_length = h_local.size(-1)

            assert local_length >= length, "Upsampling scale is small."

            h_local = F.pad(h_local, (0, length + padding - local_length))
            h_local = h_local.view(
                batch_size, local_dim, (length + padding) // num_groups, num_groups
            )
            h_local = h_local.permute(0, 1, 3, 2).contiguous()
            h_local = h_local.view(
                batch_size, local_dim * num_groups, (length + padding) // num_groups
            )

        h_global = global_conditioning

        if reverse:
            output = self._inference(
                x,
                local_conditioning=h_local,
                global_conditioning=h_global,
                logdet=logdet,
            )
        else:
            output = self._forward(
                x,
                local_conditioning=h_local,
                global_conditioning=h_global,
                logdet=logdet,
            )

        if return_logdet:
            output, logdet = output

        output = output.view(batch_size, in_channels, num_groups, (length + padding) // num_groups)
        output = output.permute(0, 1, 3, 2).contiguous()
        output = output.view(batch_size, in_channels, length + padding)

        if return_logdet:
            return output, logdet
        else:
            return output

    @torch.no_grad()
    def inference(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        std: Union[float, torch.Tensor] = 1,
        logdet: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference of WaveGlow.

        Args:
            input (torch.Tensor): Noise tensor of shape (batch_size, in_channels, length).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_dim) or (batch_size, global_dim, 1).
            std (float or torch.Tensor): Standard deviation to scale input. Default: 1.

        Returns:
            tuple of torch.Tensor.

        """
        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        output = self.forward(
            std * input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
            logdet=logdet,
            reverse=True,
        )

        return output

    def _forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of WaveGlow.

        Args:
            input (torch.Tensor): 3D tensor of shape
                (batch_size, in_channels * num_groups, length // num_groups).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, length // num_groups)).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_dim) or (batch_size, global_dim, 1).

        Returns:
            tuple of torch.Tensor.

        """
        num_flows = self.num_flows
        early_size = self.early_size

        num_features = input.size(1)
        return_logdet = logdet is not None

        x = input
        h_local = local_conditioning
        h_global = global_conditioning

        output = []

        for flow_idx in range(num_flows):
            x = self.backbone[flow_idx](
                x,
                local_conditioning=h_local,
                global_conditioning=h_global,
                logdet=logdet,
                reverse=False,
            )
            if return_logdet:
                x, logdet = x

            if flow_idx < num_flows - 1:
                x_output, x = torch.split(
                    x,
                    [early_size, num_features - (flow_idx + 1) * early_size],
                    dim=1,
                )
                output.append(x_output)
            else:
                output.append(x)

        output = torch.cat(output, dim=1)

        if return_logdet:
            return output, logdet
        else:
            return output

    def _inference(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
    ) -> torch.Tensor:
        """Reverse pass of WaveGlow.

        Args:
            input (torch.Tensor): 3D tensor of shape
                (batch_size, in_channels * num_groups, length // num_groups).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, length // num_groups)).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_dim) or (batch_size, global_dim, 1).

        Returns:
            tuple of torch.Tensor.

        """
        num_flows = self.num_flows
        early_size = self.early_size

        num_features = input.size(1)
        return_logdet = logdet is not None

        x_input = input
        h_local = local_conditioning
        h_global = global_conditioning

        for flow_idx in range(num_flows - 1, -1, -1):
            if flow_idx == num_flows - 1:
                in_channels = num_features - (num_flows - 1) * early_size
                x_input, x_stack = torch.split(
                    x_input, [num_features - in_channels, in_channels], dim=1
                )
            else:
                x_input, x = torch.split(x_input, [flow_idx * early_size, early_size], dim=1)
                x_stack = torch.cat([x, x_stack], dim=1)

            x_stack = self.backbone[flow_idx](
                x_stack,
                local_conditioning=h_local,
                global_conditioning=h_global,
                logdet=logdet,
                reverse=True,
            )
            if return_logdet:
                x_stack, logdet = x_stack

        output = x_stack

        if return_logdet:
            return output, logdet
        else:
            return output

    def weight_norm_(self) -> None:
        for flow in self.backbone:
            flow.weight_norm_()

    def remove_weight_norm_(self) -> None:
        for flow in self.backbone:
            flow.remove_weight_norm_()


class MultiSpeakerWaveGlow(WaveGlow):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_flows: int = 3,
        num_stacks: int = 4,
        num_layers: int = 8,
        num_groups: int = 8,
        early_size: int = 2,
        kernel_size: int = 3,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        upsample: Optional[nn.Module] = None,
        speaker_encoder: nn.Module = None,
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        # Use nn.Module.__init__ just for readability of
        # print(MultiSpeakerWaveGlow(...))
        super(WaveGlow, self).__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.num_flows = num_flows
        self.num_groups = num_groups
        self.early_size = early_size
        self.local_dim = local_dim

        if local_dim is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."
            local_dim = local_dim * num_groups

        self.speaker_encoder = speaker_encoder
        self.upsample = upsample

        backbone = []

        for flow_idx in range(num_flows):
            num_features = in_channels * num_groups - flow_idx * early_size

            backbone.append(
                StackedWaveGlowBlock(
                    num_features,
                    hidden_channels,
                    skip_channels=skip_channels,
                    num_stacks=num_stacks,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    bias=bias,
                    causal=causal,
                    conv=conv,
                    local_dim=local_dim,
                    global_dim=global_dim,
                    weight_norm=weight_norm,
                )
            )

        self.backbone = nn.ModuleList(backbone)

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        speaker: torch.Tensor = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> torch.Tensor:
        """Forward pass of MultiSpeakerWaveGlow.

        Args:
            input (torch.Tensor): 3D tensor of shape (batch_size, in_channels, length).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, local_length).
            speaker (torch.Tensor): Speaker feature passed to self.speaker_encoder. Usually,
                this is speaker index of shape (batch_size,), but other shapes supported
                by self.speaker_encoder can be specified.

        Returns:
            tuple of torch.Tensor.

        """
        global_conditioning = self._transform_speaker(speaker)

        output = super().forward(
            input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
            logdet=logdet,
            reverse=reverse,
        )

        return output

    @torch.no_grad()
    def inference(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        speaker: torch.Tensor = None,
        std: Union[float, torch.Tensor] = 1,
        logdet: torch.Tensor = None,
    ) -> torch.Tensor:
        """Inference of MultiSpeakerWaveGlow.

        Args:
            input (torch.Tensor): Noise tensor of shape (batch_size, in_channels, length).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_dim, local_length).
            speaker (torch.Tensor): Speaker feature passed to self.speaker_encoder. Usually,
                this is speaker index of shape (batch_size,), but other shapes supported
                by self.speaker_encoder can be specified.
            std (float or torch.Tensor): Standard deviation to scale input. Default: 1.

        Returns:
            tuple of torch.Tensor.

        """
        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        output = self.forward(
            std * input,
            local_conditioning=local_conditioning,
            speaker=speaker,
            logdet=logdet,
            reverse=True,
        )

        return output

    def _transform_speaker(
        self,
        speaker: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Transform speaker into global conditioning.

        Args:
            speaker (torch.Tensor): Speaker feature passed to self.speaker_encoder. Usually,
                this is speaker index of shape (batch_size,), but other shapes supported
                by self.speaker_encoder can be specified.

        Returns:
            torch.Tensor: Global conditioning. Its shape is determined by self.speaker_encoder.

        """
        if speaker is None:
            global_conditioning = None
        elif self.speaker_encoder is None:
            global_conditioning = speaker
        else:
            global_conditioning = self.speaker_encoder(speaker)

        return global_conditioning


class StackedWaveGlowBlock(BaseFlow):
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_stacks: int = 4,
        num_layers: int = 8,
        kernel_size: int = 3,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
    ) -> None:
        super().__init__()

        self.num_stacks = num_stacks

        backbone = []

        for _ in range(num_stacks):
            block = WaveGlowBlock(
                in_channels,
                hidden_channels,
                skip_channels=skip_channels,
                num_layers=num_layers,
                kernel_size=kernel_size,
                bias=bias,
                causal=causal,
                conv=conv,
                local_dim=local_dim,
                global_dim=global_dim,
                weight_norm=weight_norm,
                split=split,
                concat=concat,
                scaling=scaling,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        num_stacks = self.num_stacks

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)
        return_logdet = logdet is not None

        x = input

        for stack_idx in range(num_stacks):
            if reverse:
                stack_idx = num_stacks - 1 - stack_idx

            x = self.backbone[stack_idx](
                x,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

        output = x

        if return_logdet:
            return output, logdet
        else:
            return output

    def weight_norm_(self) -> None:
        for stack in self.backbone:
            stack.weight_norm_()

    def remove_weight_norm_(self) -> None:
        for stack in self.backbone:
            stack.remove_weight_norm_()


class WaveGlowBlock(BaseFlow):
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 8,
        kernel_size: int = 3,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
    ) -> None:
        super().__init__()

        if type(in_channels) is list:
            num_features = sum(in_channels)
            coupling_channels = in_channels[0]
        else:
            num_features = in_channels
            coupling_channels = in_channels // 2

        self.conv1d = InvertiblePointwiseConv1d(num_features)
        self.affine_coupling = WaveNetAffineCoupling(
            coupling_channels,
            hidden_channels,
            skip_channels=skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            bias=bias,
            causal=causal,
            conv=conv,
            local_dim=local_dim,
            global_dim=global_dim,
            weight_norm=weight_norm,
            split=split,
            concat=concat,
            scaling=scaling,
            in_channels=in_channels,
        )

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert input.dim() == 3, "input is expected to be 3D tensor, but given {}D tensor.".format(
            input.dim()
        )

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)
        return_logdet = logdet is not None

        if reverse:
            x = self.affine_coupling(
                input,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
                logdet=logdet,
                reverse=reverse,
            )
            if return_logdet:
                x, logdet = x
            else:
                logdet = None

            output = self.conv1d(x, logdet=logdet, reverse=reverse)

            if return_logdet:
                output, logdet = output

        else:
            x = self.conv1d(input, logdet=logdet, reverse=reverse)

            if return_logdet:
                x, logdet = x
            else:
                logdet = None

            output = self.affine_coupling(
                x,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                output, logdet = output

        if return_logdet:
            return output, logdet
        else:
            return output

    def weight_norm_(self) -> None:
        self.affine_coupling.weight_norm_()

    def remove_weight_norm_(self) -> None:
        self.affine_coupling.remove_weight_norm_()


class WaveNetAffineCoupling(AffineCoupling):
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
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        in_channels: Optional[int] = None,
    ) -> None:
        if local_dim is None:
            warnings.warn("local_dim is not given.")

        coupling = StackedResidualConvBlock1d(
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
            local_dim=local_dim,
            global_dim=global_dim,
            weight_norm=weight_norm,
        )

        super().__init__(
            coupling,
            split=split,
            concat=concat,
            scaling=scaling,
            in_channels=in_channels,
        )

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert input.dim() == 3, "input is expected to be 3D tensor, but given {}D tensor.".format(
            input.dim()
        )

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if reverse:
            y1, y2 = self.call_split(input)

            log_s, t = self.coupling(
                y1,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            x1 = y1
            x2 = (y2 - t) * torch.exp(-log_s)
            output = self.call_concat(x1, x2)

            if logdet is not None:
                logdet = logdet - log_s.sum(dim=(1, 2))
        else:
            x1, x2 = self.call_split(input)
            log_s, t = self.coupling(
                x1,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            y1 = x1
            y2 = x2 * torch.exp(log_s) + t
            output = self.call_concat(y1, y2)

            if logdet is not None:
                logdet = logdet + log_s.sum(dim=(1, 2))

        if logdet is None:
            return output
        else:
            return output, logdet

    def call_split(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split is None:
            num_features = input.size(1)
            output = torch.split(
                input, [num_features // 2, num_features - num_features // 2], dim=1
            )
        else:
            output = super().split(input)

        return output

    def call_concat(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.concat is None:
            output = torch.cat([input, other], dim=1)
        else:
            output = self.concat(input, other)

        return output

    def weight_norm_(self) -> None:
        self.coupling.weight_norm_()

    def remove_weight_norm_(self) -> None:
        self.coupling.remove_weight_norm_()


class StackedResidualConvBlock1d(nn.Module):
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
        local_dim: Optional[int] = None,
        global_dim: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

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
            if dilated:
                dilation = 2**layer_idx
            else:
                dilation = 1
                assert (
                    stride == 1
                ), "When dilated convolution, stride is expected to be 1, but {} is given.".format(
                    stride
                )

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
                    causal=causal,
                    dual_head=dual_head,
                    conv=conv,
                    local_dim=local_dim,
                    global_dim=global_dim,
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

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.bottleneck_conv1d_out.weight.data)
        nn.init.zeros_(self.bottleneck_conv1d_out.bias.data)

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        in_channels = self.in_channels

        x = self.bottleneck_conv1d_in(input)
        skip_connection = 0

        for layer_idx in range(self.num_layers):
            x, skip = self.backbone[layer_idx](
                x,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
            skip_connection = skip_connection + skip

        output = self.bottleneck_conv1d_out(skip_connection)
        log_s, t = torch.split(output, [in_channels, in_channels], dim=1)

        return log_s, t

    def weight_norm_(self) -> None:
        self.bottleneck_conv1d_in = nn.utils.weight_norm(self.bottleneck_conv1d_in)

        if "backbone" not in self.registered_weight_norms:
            for layer in self.backbone:
                layer.weight_norm_()

            self.registered_weight_norms.add("backbone")

    def remove_weight_norm_(self) -> None:
        self.bottleneck_conv1d_in = nn.utils.remove_weight_norm(self.bottleneck_conv1d_in)

        for layer in self.backbone:
            layer.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")
