from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from packaging import version

from .flow import AffineCoupling
from .wavenet import ResidualConvBlock1d

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class WaveNetAffineCoupling(AffineCoupling):
    def __init__(
        self,
        coupling_channels: int,
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
        scaling_channels: Optional[int] = None,
    ) -> None:
        coupling = StackedResidualConvBlock1d(
            coupling_channels,
            hidden_channels,
            skip_channels=skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            stride=1,
            dilation_rate=dilation_rate,
            bias=bias,
            is_causal=is_causal,
            conv_type=conv_type,
            local_channels=local_channels,
            global_channels=global_channels,
            weight_norm=weight_norm,
        )

        super().__init__(
            coupling,
            split=split,
            concat=concat,
            scaling=scaling,
            scaling_channels=scaling_channels,
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
                scale = scale.unsqueeze(dim=-1)
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
                scale = scale.unsqueeze(dim=-1)
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
        dilation_rate: int = 2,
        bias: bool = True,
        is_causal: bool = False,
        conv_type: str = "gated",
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
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
                    conv_type=conv_type,
                    local_channels=local_channels,
                    global_channels=global_channels,
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
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.bottleneck_conv1d_in = weight_norm_fn(self.bottleneck_conv1d_in)

        if "backbone" not in self.registered_weight_norms:
            for layer in self.backbone:
                layer.weight_norm_()

            self.registered_weight_norms.add("backbone")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.bottleneck_conv1d_in = remove_weight_norm_fn(
            self.bottleneck_conv1d_in, *remove_weight_norm_args
        )

        for layer in self.backbone:
            layer.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")
