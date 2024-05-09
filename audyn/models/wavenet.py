from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_1_t

from ..modules.wavenet import ResidualConvBlock1d

__all__ = ["WaveNet", "MultiSpeakerWaveNet"]

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class WaveNet(nn.Module):
    """WaveNet proposed in [#van2016wavenet]_.

    Args:
        in_channels (int): Number of input channels, which is typically same as out_channels.
        out_channels (int): Number of output channels.
        hidden_channels (int): Number of hidden channels in backbone.
        skip_channels (int): Number of channels in skip connection.
        num_layers (int): Number of layers. Dilation is ranged in [1, 2**(num_layers - 1)].
        num_stacks (int): Number of stacks.
        num_post_layers (int): Number of layers in post network.
        kernel_size (int): Kernel size in convolution.
        dilated (bool): Whether to apply dilated convolution.
        bias (bool): If ``True``, ``bias`` is used in convolutions.
        is_causal (bool): If ``True``, causality is ensured in convolutions.
        conv_type (str): Convolution type.
        upsample (nn.Module): Module to upsample conditional feature.
        local_channels (int): Number of channels in local conditioning.
        global_channels (int): Number of channels in global conditioning.
        weight_norm (bool): Whether to apply weight normalization.

    .. [#van2016wavenet]
        A. Oord et al., "WaveNet: A generative model for raw audio,"
        *arXiv preprint arXiv:1609.03499*, vol. 568, 2016.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 10,
        num_stacks: int = 3,
        num_post_layers: int = 2,
        kernel_size: int = 3,
        dilated: bool = True,
        bias: bool = True,
        is_causal: bool = True,
        conv_type: str = "gated",
        upsample: Optional[nn.Module] = None,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_stacks = num_stacks

        if local_channels is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."

        self.upsample = upsample

        self.causal_conv1d = nn.Conv1d(
            in_channels, hidden_channels, kernel_size=1, stride=1, dilation=1, bias=bias
        )

        backbone = []

        for stack_idx in range(num_stacks):
            if stack_idx == num_stacks - 1:
                dual_head = False
            else:
                dual_head = True

            backbone.append(
                StackedResidualConvBlock1d(
                    hidden_channels,
                    hidden_channels,
                    skip_channels=skip_channels,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    dilated=dilated,
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
        self.post_net = PostNet(
            skip_channels,
            out_channels,
            num_layers=num_post_layers,
            weight_norm=weight_norm,
        )

        # registered_weight_norms manages normalization status of backbone and post_net
        self.registered_weight_norms = set()

        if weight_norm:
            self.registered_weight_norms.add("backbone")
            self.registered_weight_norms.add("post_net")
            self.weight_norm_()

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of WaveNet.

        Args:
            input (torch.Tensor): Input tensor like waveform. Two types of input are
                supported.
                1) discrete long type input (batch_size, num_frames).
                2) continuous float type input (batch_size, in_channels, num_frames).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, num_local_frames).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, num_frames).

        """
        if local_conditioning is None:
            h_local = None
        else:
            num_timesteps = input.size(-1)
            h_local = self.upsample(local_conditioning)
            num_local_time_steps = h_local.size(-1)

            assert num_local_time_steps >= num_timesteps, "Upsampling scale is small."

            h_local = F.pad(h_local, (0, num_timesteps - num_local_time_steps))

        h_global = global_conditioning

        x = self.transform_input(input)
        x = self.causal_conv1d(x)
        residual = 0

        for stack_idx in range(self.num_stacks):
            x, skip = self.backbone[stack_idx](
                x, local_conditioning=h_local, global_conditioning=h_global
            )
            residual = residual + skip

        output = self.post_net(residual)

        return output

    @torch.no_grad()
    def inference(
        self,
        initial_state: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        max_length: int = None,
    ) -> torch.Tensor:
        """Inference of WaveNet.

        Args:
            initial_state (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, 1).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, max_length).

        """
        if max_length is None:
            if local_conditioning is None:
                raise ValueError("max_length is not specified.")
            else:
                upsampled = self.upsample(local_conditioning)
                max_length = upsampled.size(-1)

        in_channels, out_channels = self.in_channels, self.out_channels
        batch_size, length = initial_state.size(0), initial_state.size(-1)
        dim = initial_state.dim()

        if length != 1:
            raise ValueError("Length of initial_state must be 1, but given {}.".format(length))

        factory_kwargs = {
            "dtype": initial_state.dtype,
            "device": initial_state.device,
        }

        if dim == 2:
            if initial_state.dtype not in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ]:
                raise TypeError(
                    "torch.LongTensor is expected as initial_state, but {} is given.".format(
                        type(initial_state)
                    )
                )
            incremental_buffered_output = torch.zeros((batch_size, 0), **factory_kwargs)
        elif dim == 3:
            if initial_state.dtype not in [torch.float16, torch.float32, torch.float64]:
                raise TypeError(
                    "torch.FloatTensor is expected as initial_state, but {} is given.".format(
                        type(initial_state)
                    )
                )
            if out_channels != in_channels:
                raise ValueError(
                    "out_channels {} != in_channels {}.".format(out_channels, in_channels)
                )
            incremental_buffered_output = torch.zeros(
                (batch_size, out_channels, 0), **factory_kwargs
            )
        else:
            raise ValueError(
                "Only 2 and 3D are supported as initial_state, but given {}D.".format(dim)
            )

        self.clear_buffer()

        last_output = initial_state

        for _ in range(max_length):
            last_output = self.incremental_forward(
                last_output,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )

            if dim == 2:
                # sampling from categorical distribution
                last_output = torch.softmax(last_output, dim=1)
                last_output = last_output.permute(0, 2, 1)
                last_output = torch.distributions.Categorical(last_output).sample()

            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        self.clear_buffer()

        return incremental_buffered_output

    def incremental_forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Incremental forward pass of WaveNet.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, 1).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, num_local_frames).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, 1).

        """
        if local_conditioning is None:
            h_local = None
        else:
            if not hasattr(self, "local_buffer"):
                self.local_buffer = None

            if self.local_buffer is None:
                local_buffer = self.upsample(local_conditioning)
            else:
                # Since local_conditioning is saved as local_buffer,
                # we don't use local_conditioning argument here.
                local_buffer = self.local_buffer

            num_local_time_steps = local_buffer.size(-1)
            h_local, local_buffer = torch.split(
                local_buffer, [1, num_local_time_steps - 1], dim=-1
            )
            self.local_buffer = local_buffer

        h_global = global_conditioning

        x = self.transform_input(input)
        x = self.causal_conv1d(x)
        residual = 0

        for stack_idx in range(self.num_stacks):
            stack: StackedResidualConvBlock1d = self.backbone[stack_idx]
            x, skip = stack.incremental_forward(
                x, local_conditioning=h_local, global_conditioning=h_global
            )
            residual = residual + skip

        output = self.post_net(residual)

        return output

    def clear_buffer(self) -> None:
        if hasattr(self, "local_buffer"):
            self.local_buffer = None

        for module in self.modules():
            if module is self:
                continue

            if hasattr(module, "clear_buffer") and callable(module.clear_buffer):
                module.clear_buffer()

    def transform_input(self, input: torch.Tensor) -> torch.Tensor:
        dim = input.dim()

        if dim == 2:
            if input.dtype not in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ]:
                raise TypeError(
                    "torch.LongTensor is expected as input, but {} is given.".format(type(input))
                )
            x = F.one_hot(input, num_classes=self.in_channels)
            output = x.permute(0, 2, 1).float()
        elif dim == 3:
            if input.dtype not in [torch.float16, torch.float32, torch.float64]:
                raise TypeError(
                    "torch.FloatTensor is expected as input, but {} is given.".format(type(input))
                )
            output = input
        else:
            raise ValueError("Only 2 and 3D inputs are supported, but given {}D.".format(dim))

        return output

    def weight_norm_(self) -> None:
        """Applies weight normalization to self.causal_conv1d.

        .. note::

            This method applies normalizations to self.backbone
            and self.post_net as well if necessary.

        """
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.causal_conv1d = weight_norm_fn(self.causal_conv1d)

        if "backbone" not in self.registered_weight_norms:
            for module in self.backbone:
                module.weight_norm_()

            self.registered_weight_norms.add("backbone")

        if "post_net" not in self.registered_weight_norms:
            self.post_net.weight_norm_()
            self.registered_weight_norms.add("post_net")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.causal_conv1d = remove_weight_norm_fn(self.causal_conv1d, *remove_weight_norm_args)

        for module in self.backbone:
            module.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.post_net.remove_weight_norm_()
        self.registered_weight_norms.remove("post_net")


class MultiSpeakerWaveNet(WaveNet):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 10,
        num_stacks: int = 3,
        num_post_layers: int = 2,
        kernel_size: int = 3,
        dilated: bool = True,
        bias: bool = True,
        is_causal: bool = True,
        conv_type: str = "gated",
        upsample: Optional[nn.Module] = None,
        speaker_encoder: nn.Module = None,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        # Use nn.Module.__init__ just for readability of
        # print(MultiSpeakerWaveNet(...))
        super(WaveNet, self).__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels, self.out_channels = in_channels, out_channels
        self.num_stacks = num_stacks

        if local_channels is None:
            assert upsample is None, "upsample is expected to None."
        else:
            assert upsample is not None, "upsample is expected to be given."

        self.speaker_encoder = speaker_encoder
        self.upsample = upsample

        self.causal_conv1d = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            bias=bias,
        )

        backbone = []

        for stack_idx in range(num_stacks):
            if stack_idx == num_stacks - 1:
                dual_head = False
            else:
                dual_head = True

            backbone.append(
                StackedResidualConvBlock1d(
                    hidden_channels,
                    hidden_channels,
                    skip_channels=skip_channels,
                    num_layers=num_layers,
                    kernel_size=kernel_size,
                    dilated=dilated,
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
        self.post_net = PostNet(
            skip_channels,
            out_channels,
            num_layers=num_post_layers,
            weight_norm=weight_norm,
        )

        # registered_weight_norms manages normalization status of backbone and post_net
        self.registered_weight_norms = set()

        if weight_norm:
            self.registered_weight_norms.add("backbone")
            self.registered_weight_norms.add("post_net")
            self.weight_norm_()

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        speaker: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass of MultiSpeakerWaveNet.

        Args:
            input (torch.Tensor): Input tensor like waveform. Two types of input are
                supported.
                1) discrete long type input (batch_size, num_frames).
                2) continuous float type input (batch_size, in_channels, num_frames).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, num_local_frames).
            speaker (torch.Tensor): Speaker feature passed to self.speaker_encoder. Usually,
                this is speaker index of shape (batch_size,), but other shapes supported
                by self.speaker_encoder can be specified.

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, num_frames).

        """
        global_conditioning = self._transform_speaker(speaker)

        output = super().forward(
            input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
        )

        return output

    @torch.no_grad()
    def inference(
        self,
        initial_state: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        speaker: torch.Tensor = None,
        max_length: int = None,
    ) -> torch.Tensor:
        """Inference of MultiSpeakerWaveNet.

        Args:
            initial_state (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, 1).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, local_length).
            speaker (torch.Tensor): Speaker feature passed to self.speaker_encoder. Usually,
                this is speaker index of shape (batch_size,), but other shapes supported
                by self.speaker_encoder can be specified.

        Returns:
            torch.Tensor: Output of shape (batch_size, out_channels, max_length).

        """
        global_conditioning = self._transform_speaker(speaker)

        incremental_buffered_output = super().inference(
            initial_state,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
            max_length=max_length,
        )

        return incremental_buffered_output

    def incremental_forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        *,
        speaker: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if speaker is not None and global_conditioning is not None:
            raise ValueError(
                "At least, one of speaker and global_conditioning should be None.",
            )

        if speaker is not None:
            global_conditioning = self._transform_speaker(speaker)

        return super().incremental_forward(
            input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
        )

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


class StackedResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 10,
        kernel_size: int = 3,
        dilated: bool = True,
        bias: bool = True,
        is_causal: bool = True,
        dual_head: bool = True,
        conv_type: str = "gated",
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.num_layers = num_layers

        net = []

        for layer_idx in range(num_layers):
            if dilated:
                dilation = 2**layer_idx
            else:
                dilation = 1

            if layer_idx < num_layers - 1 or dual_head:
                _dual_head = True
            else:
                _dual_head = False

            net.append(
                ResidualConvBlock1d(
                    in_channels,
                    hidden_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation,
                    bias=bias,
                    is_causal=is_causal,
                    dual_head=_dual_head,
                    conv_type=conv_type,
                    local_channels=local_channels,
                    global_channels=global_channels,
                    weight_norm=weight_norm,
                )
            )

        self.net = nn.ModuleList(net)

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Forward pass of stacked residual convolution.

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
        x = input
        skip_connection = 0

        for layer_idx in range(self.num_layers):
            x, skip = self.net[layer_idx](
                x,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
            skip_connection = skip_connection + skip

        output = x

        return output, skip_connection

    def incremental_forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        assert input.size(-1) == 1

        if local_conditioning is not None:
            assert local_conditioning.size(-1) == 1

        x = input
        skip_connection = 0

        for layer_idx in range(self.num_layers):
            layer: ResidualConvBlock1d = self.net[layer_idx]
            x, skip = layer.incremental_forward(
                x,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
            skip_connection = skip_connection + skip

        output = x

        return output, skip_connection

    def weight_norm_(self) -> None:
        for module in self.net:
            module.weight_norm_()

    def remove_weight_norm_(self) -> None:
        for module in self.net:
            module.remove_weight_norm_()


class PostNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        net = []

        for layer_idx in range(num_layers):
            if layer_idx == num_layers - 1:
                _out_channels = out_channels
            else:
                _out_channels = in_channels

            net.append(
                PostBlock(
                    in_channels,
                    _out_channels,
                    weight_norm=weight_norm,
                )
            )

        self.net = nn.ModuleList(net)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        for layer_idx in range(self.num_layers):
            x = self.net[layer_idx](x)

        output = x

        return output

    def weight_norm_(self) -> None:
        for module in self.net:
            module.weight_norm_()

    def remove_weight_norm_(self) -> None:
        for module in self.net:
            module.remove_weight_norm_()


class PostBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        self.nonlinear1d = nn.ReLU()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1)

        if weight_norm:
            self.weight_norm_()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.nonlinear1d(input)
        output = self.conv1d(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv1d = weight_norm_fn(self.conv1d)

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv1d = remove_weight_norm_fn(self.conv1d, *remove_weight_norm_args)


class Upsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.stride = stride

        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of upsample network.

        Args:
            input (torch.Tensor): 3D tensor of shape (batch_size, in_channels, length).

        Returns:
            output (torch.Tensor): 3D tensor of shape (batch_size, out_channels, length // stride).

        """
        stride = self.stride
        length = input.size(-1)

        x = self.conv_transpose1d(input)
        padding = x.size(-1) - length * stride
        padding_left = padding // 2
        padding_right = padding - padding_left
        output = F.pad(x, (-padding_left, -padding_right))

        return output
