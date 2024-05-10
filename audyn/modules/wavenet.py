from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class ResidualConvBlock1d(nn.Module):
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
        conv_type: str = "gated",
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ) -> None:
        super().__init__()

        if skip_channels is None:
            skip_channels = hidden_channels

        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.kernel_size, self.dilation = kernel_size, dilation
        self.is_causal = is_causal
        self.dual_head = dual_head

        if conv_type == "gated":
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
            raise ValueError("{} is not supported for conv.".format(conv_type))

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
        skip = self.skip_conv1d(x)

        if self.dual_head:
            output = self.output_conv1d(x)
            output = residual + output
        else:
            output = None

        return output, skip

    def incremental_forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        is_causal = self.is_causal

        if not is_causal:
            raise NotImplementedError("Only causal forward is supported.")

        assert input.size(-1) == 1

        if local_conditioning is not None:
            assert local_conditioning.size(-1) == 1

        residual = input
        x = self.conv1d.incremental_forward(
            input,
            local_conditioning=local_conditioning,
            global_conditioning=global_conditioning,
        )
        skip = self.skip_conv1d(x)

        if self.dual_head:
            output = self.output_conv1d(x)
            output = residual + output
        else:
            output = None

        return output, skip

    def weight_norm_(self) -> None:
        """Applies weight normalizations to self.output_conv1d and self.skip_conv1d.

        .. note::

            This method applies normalization to self.conv1d
            as well if necessary.

        """
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        if "conv1d" not in self.registered_weight_norms:
            self.conv1d.weight_norm_()
            self.registered_weight_norms.add("conv1d")

        if self.output_conv1d is not None:
            self.output_conv1d = weight_norm_fn(self.output_conv1d)

        self.skip_conv1d = weight_norm_fn(self.skip_conv1d)

    def remove_weight_norm_(self) -> None:
        """Remove weight normalization from weights of convolution modules."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv1d.remove_weight_norm_()
        self.registered_weight_norms.remove("conv1d")

        if self.output_conv1d is not None:
            self.output_conv1d = remove_weight_norm_fn(
                self.output_conv1d, *remove_weight_norm_args
            )

        self.skip_conv1d = remove_weight_norm_fn(self.skip_conv1d, *remove_weight_norm_args)


class GatedConv1d(nn.Module):
    """Gated convolution used in WaveNet.

    Args:
        weight_norm (bool): If ``True``, weight normalization is used for weights of convolution.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        dilation: int = 1,
        bias: bool = True,
        is_causal: bool = True,
        local_channels: Optional[int] = None,
        global_channels: Optional[int] = None,
        weight_norm: bool = True,
    ):
        super().__init__()

        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.is_causal = is_causal

        self.conv1d_tanh = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )
        self.conv1d_sigmoid = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

        if local_channels is None:
            self.local_conv1d_tanh = None
            self.local_conv1d_sigmoid = None
        else:
            self.local_conv1d_tanh = nn.Conv1d(
                local_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )
            self.local_conv1d_sigmoid = nn.Conv1d(
                local_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )

        if global_channels is None:
            self.global_conv1d_tanh = None
            self.global_conv1d_sigmoid = None
        else:
            self.global_conv1d_tanh = nn.Conv1d(
                global_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )
            self.global_conv1d_sigmoid = nn.Conv1d(
                global_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            )

        # for weight normalization
        self.weight_norm_conv_names = set()

        if weight_norm:
            self.weight_norm_()

    def forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of gated convolution.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, num_frames).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, num_frames).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            torch.Tensor: Output of shape (batch_size, in_channels, num_frames).

        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        is_causal = self.is_causal

        padding = (kernel_size - 1) * dilation

        if is_causal:
            padding_left = padding
            padding_right = 0
        else:
            padding_left = padding // 2
            padding_right = padding - padding_left

        x = F.pad(input, (padding_left, padding_right))

        x_tanh, x_sigmoid = self._fused_conv1d(
            x,
            conv1d_tanh=self.conv1d_tanh,
            conv1d_sigmoid=self.conv1d_sigmoid,
            stride=stride,
            dilation=dilation,
        )

        if local_conditioning is not None:
            assert (self.local_conv1d_sigmoid is not None) and (self.local_conv1d_tanh is not None)

            y_tanh, y_sigmoid = self._fused_conv1d(
                local_conditioning,
                conv1d_tanh=self.local_conv1d_tanh,
                conv1d_sigmoid=self.local_conv1d_sigmoid,
                stride=1,
            )

            x_tanh = x_tanh + y_tanh
            x_sigmoid = x_sigmoid + y_sigmoid

        if global_conditioning is not None:
            assert (self.global_conv1d_sigmoid is not None) and (
                self.global_conv1d_tanh is not None
            )

            if global_conditioning.dim() == 2:
                global_conditioning = global_conditioning.unsqueeze(dim=-1)

            y_tanh, y_sigmoid = self._fused_conv1d(
                global_conditioning,
                conv1d_tanh=self.global_conv1d_tanh,
                conv1d_sigmoid=self.global_conv1d_sigmoid,
                stride=1,
            )

            x_tanh = x_tanh + y_tanh
            x_sigmoid = x_sigmoid + y_sigmoid

        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)

        output = x_tanh * x_sigmoid

        return output

    def incremental_forward(
        self,
        input: torch.Tensor,
        local_conditioning: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of gated convolution.

        Args:
            input (torch.Tensor): Input tensor of shape
                (batch_size, in_channels, 1).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, 1).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            torch.Tensor: Output of shape (batch_size, in_channels, 1).

        .. note:

            In this method, ``self._forward_pre_hooks`` is called only if ``self.buffer``
            is ``None``, i.e. first call of ``incremental_forward``.

        """
        kernel_size, stride, dilation = self.kernel_size, self.stride, self.dilation
        is_causal = self.is_causal

        if not is_causal:
            raise NotImplementedError("Only causal forward is supported.")

        assert input.size(-1) == 1

        if local_conditioning is not None:
            assert local_conditioning.size(-1) == 1

        padding = (kernel_size - 1) * dilation

        if not hasattr(self, "buffer"):
            self.buffer = None

        if self.buffer is None:
            batch_size, in_channels, _ = input.size()
            buffer = torch.zeros(
                (batch_size, in_channels, padding), dtype=input.dtype, device=input.device
            )
            for hook in self._forward_pre_hooks.values():
                hook(self, input)
        else:
            buffer = self.buffer
            _, buffer = torch.split(buffer, [1, padding], dim=-1)

        buffer = torch.cat([buffer, input], dim=-1)
        self.buffer = buffer

        x_tanh, x_sigmoid = self._fused_conv1d(
            buffer,
            conv1d_tanh=self.conv1d_tanh,
            conv1d_sigmoid=self.conv1d_sigmoid,
            stride=stride,
            dilation=dilation,
        )

        if local_conditioning is not None:
            assert (self.local_conv1d_sigmoid is not None) and (self.local_conv1d_tanh is not None)

            if not hasattr(self, "local_buffer"):
                self.local_buffer = None

            if self.local_buffer is None:
                batch_size, local_channels, _ = local_conditioning.size()
                local_buffer = torch.zeros(
                    (batch_size, local_channels, 0),
                    dtype=local_conditioning.dtype,
                    device=local_conditioning.device,
                )
            else:
                local_buffer = self.local_buffer

            local_buffer = torch.cat([local_buffer, local_conditioning], dim=-1)
            self.local_buffer = local_buffer

            y_tanh, y_sigmoid = self._fused_conv1d(
                local_conditioning,
                conv1d_tanh=self.local_conv1d_tanh,
                conv1d_sigmoid=self.local_conv1d_sigmoid,
                stride=1,
            )

            x_tanh = x_tanh + y_tanh
            x_sigmoid = x_sigmoid + y_sigmoid

        if global_conditioning is not None:
            assert (self.global_conv1d_sigmoid is not None) and (
                self.global_conv1d_tanh is not None
            )

            if global_conditioning.dim() == 2:
                global_conditioning = global_conditioning.unsqueeze(dim=-1)

            y_tanh, y_sigmoid = self._fused_conv1d(
                global_conditioning,
                conv1d_tanh=self.global_conv1d_tanh,
                conv1d_sigmoid=self.global_conv1d_sigmoid,
                stride=1,
            )

            x_tanh = x_tanh + y_tanh
            x_sigmoid = x_sigmoid + y_sigmoid

        x_tanh = torch.tanh(x_tanh)
        x_sigmoid = torch.sigmoid(x_sigmoid)

        output = x_tanh * x_sigmoid

        return output

    def clear_buffer(self):
        if hasattr(self, "buffer"):
            self.buffer = None

        if hasattr(self, "local_buffer"):
            self.local_buffer = None

    def weight_norm_(self) -> None:
        """Apply weight normalization to weights of convolution modules."""
        if self.weight_norm_registered:
            raise ValueError(
                "Weight normalization is already applied."
                "Call remove_weight_norm_() before this method."
            )

        GatedConv1dWeightNorm.apply(self, conv_name="conv1d_tanh")
        GatedConv1dWeightNorm.apply(self, conv_name="conv1d_sigmoid")
        self.weight_norm_conv_names.add("conv1d_tanh")
        self.weight_norm_conv_names.add("conv1d_sigmoid")

        if self.local_conv1d_tanh is not None:
            GatedConv1dWeightNorm.apply(self, conv_name="local_conv1d_tanh")
            GatedConv1dWeightNorm.apply(self, conv_name="local_conv1d_sigmoid")
            self.weight_norm_conv_names.add("local_conv1d_tanh")
            self.weight_norm_conv_names.add("local_conv1d_sigmoid")

        if self.global_conv1d_tanh is not None:
            GatedConv1dWeightNorm.apply(self, conv_name="global_conv1d_tanh")
            GatedConv1dWeightNorm.apply(self, conv_name="global_conv1d_sigmoid")
            self.weight_norm_conv_names.add("global_conv1d_tanh")
            self.weight_norm_conv_names.add("global_conv1d_sigmoid")

    def remove_weight_norm_(self) -> None:
        """Remove weight normalization from weights of convolution modules."""
        if len(self.weight_norm_conv_names) == 0:
            raise ValueError("weight_norm of is not found.")

        pre_hook_keys = list(self._forward_pre_hooks.keys())

        for k in pre_hook_keys:
            hook = self._forward_pre_hooks[k]
            if (
                isinstance(hook, GatedConv1dWeightNorm)
                and hook.conv_name in self.weight_norm_conv_names
            ):
                hook.remove(self)
                del self._forward_pre_hooks[k]
                self.weight_norm_conv_names.remove(hook.conv_name)

        if len(self.weight_norm_conv_names) > 0:
            raise ValueError(f"weight_norms of {self.weight_norm_conv_names} cannot be removed.")

    @staticmethod
    def _fused_conv1d(
        input: torch.Tensor,
        conv1d_tanh: nn.Conv1d,
        conv1d_sigmoid: nn.Conv1d,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        weight_tanh = conv1d_tanh.weight
        weight_sigmoid = conv1d_sigmoid.weight
        weight = torch.cat([weight_tanh, weight_sigmoid], dim=0)
        out_channels_tanh = weight_tanh.size(0)
        out_channels_sigmoid = weight_sigmoid.size(0)

        assert (conv1d_tanh.bias is None) == (conv1d_sigmoid.bias is None)

        if conv1d_tanh.bias is None:
            bias = None
        else:
            bias = torch.cat([conv1d_tanh.bias, conv1d_sigmoid.bias], dim=0)

        output = F.conv1d(
            input,
            weight,
            bias=bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        output_tanh, output_sigmoid = torch.split(
            output, [out_channels_tanh, out_channels_sigmoid], dim=1
        )

        return output_tanh, output_sigmoid

    @property
    def weight_norm_registered(self) -> bool:
        return len(self.weight_norm_conv_names) > 0


class GatedConv1dWeightNorm:
    """Weight normalization for module containing GatedConv1d.

    This implementation is based on torch.nn.utils.weight_norm.WeightNorm.

    Args:
        conv_name (str): Name of gated convolution.
        dim (int): Normalization is applied along ``dim``.

    """

    def __init__(self, conv_name: str) -> None:
        self.conv_name = conv_name
        self.name = "weight"
        self.dim = 0

    def __call__(self, module: nn.Module, inputs: Any) -> None:
        conv_module = getattr(module, self.conv_name)
        setattr(conv_module, self.name, self.compute_weight(conv_module))

    @staticmethod
    def apply(module: nn.Module, conv_name: str) -> None:
        """Apply weight normalization to convolution in module.

        Reference of implementation:
        https://github.com/pytorch/pytorch/blob/31f311a816c026bbfca622d6121d6a7fab44260d/torch/nn/utils/weight_norm.py

        Args:
            module (nn.Module): Module instance to which normalization is applied.
            conv_name (str): Name of gated convolution.

        """
        conv_module: GatedConv1d = getattr(module, conv_name)

        fn = GatedConv1dWeightNorm(conv_name)
        name = fn.name
        dim = fn.dim

        weight = getattr(conv_module, name)
        weight_g = torch.norm_except_dim(weight, pow=2, dim=dim).data
        weight_v = weight.data

        del conv_module._parameters[name]

        conv_module.register_parameter(f"{name}_g", nn.Parameter(weight_g))
        conv_module.register_parameter(f"{name}_v", nn.Parameter(weight_v))
        weight = fn.compute_weight(conv_module)

        setattr(conv_module, name, weight)

        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: nn.Module) -> None:
        name = self.name
        conv_module = getattr(module, self.conv_name)
        weight = self.compute_weight(conv_module)

        delattr(conv_module, name)
        del conv_module._parameters[f"{name}_g"]
        del conv_module._parameters[f"{name}_v"]

        setattr(conv_module, self.name, nn.Parameter(weight.data))

    def compute_weight(self, conv_module: GatedConv1d) -> Any:
        name = self.name
        dim = self.dim

        weight_g = getattr(conv_module, f"{name}_g")
        weight_v = getattr(conv_module, f"{name}_v")
        weight = torch._weight_norm(weight_v, weight_g, dim=dim)

        return weight
