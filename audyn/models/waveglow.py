from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.flow import BaseFlow
from ..modules.glow import InvertiblePointwiseConv1d
from ..modules.waveglow import WaveNetAffineCoupling

__all__ = [
    "WaveGlow",
    "MultiSpeakerWaveGlow",
    "WaveNetAffineCoupling",  # for backward compatibility
]


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
        dilation_rate: int = 2,
        bias: bool = True,
        is_causal: bool = False,
        conv_type: str = "gated",
        upsample: Optional[nn.Module] = None,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.num_flows = num_flows
        self.num_groups = num_groups
        self.early_size = early_size
        self.local_channels = local_channels

        if local_channels is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."
            local_channels = local_channels * num_groups

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
                    dilation_rate=dilation_rate,
                    bias=bias,
                    is_causal=is_causal,
                    conv_type=conv_type,
                    local_channels=local_channels,
                    global_channels=global_channels,
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
                (batch_size, local_channels, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            tuple of torch.Tensor.

        """
        num_groups = self.num_groups
        local_channels = self.local_channels

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
                batch_size, local_channels, (length + padding) // num_groups, num_groups
            )
            h_local = h_local.permute(0, 1, 3, 2).contiguous()
            h_local = h_local.view(
                batch_size, local_channels * num_groups, (length + padding) // num_groups
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
        output = F.pad(output, (0, -padding))

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
                (batch_size, local_channels, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).
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
                (batch_size, local_channels, length // num_groups)).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

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
                (batch_size, local_channels, length // num_groups)).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

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
        dilation_rate: int = 2,
        bias: bool = True,
        is_causal: bool = False,
        conv_type: str = "gated",
        upsample: Optional[nn.Module] = None,
        speaker_encoder: nn.Module = None,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
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
        self.local_channels = local_channels

        if local_channels is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."
            local_channels = local_channels * num_groups

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
                    dilation_rate=dilation_rate,
                    bias=bias,
                    is_causal=is_causal,
                    conv_type=conv_type,
                    local_channels=local_channels,
                    global_channels=global_channels,
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
                (batch_size, local_channels, local_length).
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
                (batch_size, local_channels, local_length).
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
        dilation_rate: int = 2,
        bias: bool = True,
        is_causal: bool = False,
        conv_type: str = "gated",
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
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
                dilation_rate=dilation_rate,
                bias=bias,
                is_causal=is_causal,
                conv_type=conv_type,
                local_channels=local_channels,
                global_channels=global_channels,
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
        dilation_rate: int = 2,
        bias: bool = True,
        is_causal: bool = False,
        conv_type: str = "gated",
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
    ) -> None:
        super().__init__()

        # TODO: support list type input properly
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
            dilation_rate=dilation_rate,
            bias=bias,
            is_causal=is_causal,
            conv_type=conv_type,
            local_channels=local_channels,
            global_channels=global_channels,
            weight_norm=weight_norm,
            split=split,
            concat=concat,
            scaling=scaling,
            scaling_channels=in_channels,
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
