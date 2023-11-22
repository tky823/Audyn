import copy
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from audyn.modules.pixelcnn import CausalConv2d as PixelCNNCausalConv2d
from audyn.modules.pixelcnn import VerticalConv2d
from audyn.modules.pixelsnail import CausalConv2d as PixelSNAILCausalConv2d
from audyn.modules.pixelsnail import _generate_square_subsequent_mask, _get_activation

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class PixelSNAIL(nn.Module):
    """PixelSNAIL used in liu2021conditional [#liu2021conditional]_.

    .. [#liu2021conditional]
        X. Liu et al.,
        "Conditional sound generation using neural discrete time-frequency representation
        learning," in **MLSP**, 2021, pp.1-6.

    """

    def __init__(
        self,
        codebook_size: int,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        num_heads: int,
        num_blocks: int,
        num_repeats: int,
        num_post_blocks: int = 0,
        dropout: float = 0.1,
        conditional_channels: Optional[int] = None,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        input_shape: Tuple[int, int] = None,
        conditionor: Optional[nn.Module] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        kernel_size = _pair(kernel_size)

        self.codebook_size = codebook_size
        self.num_blocks = num_blocks
        self.input_shape = input_shape

        background = self.create_background(input_shape)
        self.register_buffer("background", background)

        kh, kw = kernel_size
        _kernel_size = (kh + 1) // 2, kw // 2

        self.vertical_conv2d = VerticalConv2d(
            codebook_size,
            in_channels,
            kernel_size=kernel_size,
            capture_center=True,
        )
        self.horizontal_conv2d = PixelSNAILCausalConv2d(
            codebook_size,
            in_channels,
            kernel_size=_kernel_size,
        )
        self.conditionor = conditionor

        backbone = []

        for _ in range(num_blocks):
            block = PixelBlock(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                num_heads=num_heads,
                num_repeats=num_repeats,
                dropout=dropout,
                conditional_channels=conditional_channels,
                weight_regularization=weight_regularization,
                activation=activation,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)
        self.post_net = PostNet(
            in_channels,
            codebook_size,
            hidden_channels,
            num_blocks=num_post_blocks,
            weight_regularization=weight_regularization,
        )

        # registered_weight_norms manages normalization status
        self.registered_weight_norms = set()

        if weight_regularization == "weight_norm":
            self.registered_weight_norms.add("backbone")
            self.registered_weight_norms.add("post_net")
            self.weight_norm_()

    def forward(
        self,
        input: torch.LongTensor,
        conditioning: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of PixelSNAIL.

        Args:
            input (torch.LongTensor): Codebook indices of shape (batch_size, height, width).
            conditioning (torch.LongTensor): Conditional feature of shape (batch_size,).

        Returns:
            torch.Tensor (torch.Tensor): Output feature of shape (batch_size, height, width).

        """
        self.background: nn.Parameter

        codebook_size = self.codebook_size
        num_blocks = self.num_blocks
        background = self.background

        x = F.one_hot(input, num_classes=codebook_size)
        x = x.permute(0, 3, 1, 2)
        x = x.to(background.dtype)
        x_vertical = self.vertical_conv2d(x)
        x_horizontal = self.horizontal_conv2d(x)

        x_vertical = F.pad(x_vertical, (0, 0, 1, -1))
        x_horizontal = F.pad(x_horizontal, (1, -1))
        x = x_horizontal + x_vertical

        if conditioning is None:
            x_conditioning = None
        else:
            x_conditioning = self.conditionor(conditioning)

        height_in, width_in = x.size()[-2:]
        height, width = background.size()[-2:]

        assert height >= height_in and width >= width_in

        x_background = F.pad(background, (0, width_in - width, 0, height_in - height))

        if x_conditioning is not None:
            x_conditioning = F.pad(x_conditioning, (0, width_in - width, 0, height_in - height))

        for block_idx in range(num_blocks):
            x = self.backbone[block_idx](x, x_background, conditioning=x_conditioning)

        output = self.post_net(x)

        return output

    @torch.no_grad()
    def inference(
        self,
        initial_state: torch.LongTensor,
        conditioning: Optional[torch.LongTensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.LongTensor:
        self.background: nn.Parameter

        if height is None:
            height = self.input_shape[0]

        if width is None:
            width = self.input_shape[1]

        batch_size, height_in, width_in = initial_state.size()

        assert (height_in, width_in) == (1, 1)

        output = F.pad(initial_state, (0, width - 1, 0, height - 1))

        for row_idx in range(height):
            for column_idx in range(width):
                x = F.pad(output, (0, 0, 0, -(height - 1 - row_idx)))
                x = self.forward(x, conditioning=conditioning)
                last_output = F.pad(x, (-column_idx, -(width - 1 - column_idx), -row_idx, 0))
                last_output = last_output.view(batch_size, -1)

                # sampling from categorical distribution
                last_output = torch.softmax(last_output, dim=1)
                last_output = torch.distributions.Categorical(last_output).sample()
                output[:, row_idx, column_idx] = last_output

        return output

    @staticmethod
    def create_background(shape: Tuple[int, int]) -> torch.Tensor:
        height, width = shape
        x = torch.arange(width, dtype=torch.float) / width - 0.5
        y = torch.arange(height, dtype=torch.float) / height - 0.5
        x, y = torch.meshgrid(x, y, indexing="xy")
        background = torch.stack([y, x], dim=0)

        return background

    def weight_norm_(self) -> None:
        """Set weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.vertical_conv2d = weight_norm_fn(self.vertical_conv2d)
        self.horizontal_conv2d = weight_norm_fn(self.horizontal_conv2d)
        self.registered_weight_norms.add("vertical_conv2d")
        self.registered_weight_norms.add("horizontal_conv2d")

        if "backbone" not in self.registered_weight_norms:
            for block in self.backbone:
                block: PixelBlock
                block.weight_norm_()

            self.registered_weight_norms.add("backbone")

        if "post_net" not in self.registered_weight_norms:
            self.post_net.weight_norm_()
            self.registered_weight_norms.add("post_net")

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.vertical_conv2d = remove_weight_norm_fn(
            self.vertical_conv2d, *remove_weight_norm_args
        )
        self.horizontal_conv2d = remove_weight_norm_fn(
            self.horizontal_conv2d, *remove_weight_norm_args
        )

        self.registered_weight_norms.remove("vertical_conv2d")
        self.registered_weight_norms.remove("horizontal_conv2d")

        for block in self.backbone:
            block: PixelBlock
            block.weight_norm_()

        self.registered_weight_norms.remove("backbone")

        self.post_net.remove_weight_norm_()
        self.registered_weight_norms.remove("post_net")


class EmbeddingNet(nn.Module):
    """Embedding network to transform discrete input to dense feature."""

    def __init__(
        self,
        num_embeddings: int,
        shape: _size_2_t,
        pad_idx: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.shape = _pair(shape)

        embedding_dim = 1

        for s in self.shape:
            embedding_dim *= s

        self.embedding = nn.Embedding(
            num_embeddings,
            embedding_dim,
            padding_idx=pad_idx,
        )

    def forward(self, input: torch.LongTensor) -> torch.Tensor:
        """Transform discrete input to dense feature.

        Args:
            input (torch.LongTensor): (batch_size,)

        Returns:
            torch.Tensor: Embeddings of shape (batch_size, 1, height, width)

        """
        x = self.embedding(input)
        output = x.view(-1, 1, *self.shape)

        return output


class PostNet(nn.Module):
    """PostNet of PixelSNAIL."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_blocks: int,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.num_blocks = num_blocks

        backbone = []

        for _ in range(num_blocks):
            block = ResidualBlock2d(
                in_channels,
                hidden_channels,
                kernel_size=1,
                weight_regularization=weight_regularization,
            )
            backbone.append(block)

        self.backbone = nn.Sequential(*backbone)
        self.activation2d = nn.ELU()
        self.conv2d = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

        # registered_weight_norms and registered_spectral_norms manage normalization status
        self.registered_weight_norms = set()
        self.registered_spectral_norms = set()

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.registered_weight_norms.add("backbone")
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.registered_spectral_norms.add("backbone")
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PostNet.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        x = self.backbone(input)
        x = self.activation2d(x)
        output = self.conv2d(x)

        return output

    def weight_norm_(self) -> None:
        """Set weight_norm to conv2d."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d = weight_norm_fn(self.conv2d)
        self.registered_weight_norms.add("conv2d")

        if "backbone" not in self.registered_weight_norms:
            for block in self.backbone:
                block: ResidualBlock2d
                block.weight_norm_()

            self.registered_weight_norms.add("backbone")

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d = remove_weight_norm_fn(self.conv2d, *remove_weight_norm_args)
        self.registered_weight_norms.remove("conv2d")

        for block in self.backbone:
            block: ResidualBlock2d
            block.remove_weight_norm_()

        self.registered_weight_norms.remove("backbone")

    def spectral_norm_(self) -> None:
        """Set spectral_norm to conv2d."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d = spectral_norm_fn(self.conv2d)
        self.registered_spectral_norms.add("conv2d")

        if "backbone" not in self.registered_spectral_norms:
            for block in self.backbone:
                block: ResidualBlock2d
                block.spectral_norm_()

            self.registered_spectral_norms.add("backbone")

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d = remove_spectral_norm_fn(self.conv2d, *remove_spectral_norm_args)
        self.registered_spectral_norms.remove("conv2d")

        for block in self.backbone:
            block: ResidualBlock2d
            block.remove_spectral_norm_()

        self.registered_spectral_norms.remove("backbone")


class PixelBlock(nn.Module):
    """Block of PixelSNAIL.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels used in ``ResidualBlock2d``.
        kernel_size (_size_2_t): Kernel size in convolutions.
        num_heads (int): Number of heads in attention.
        num_repeats (int): Number of repeats of ``ResidualBlock2d``.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        weight_regularization (str, optional): Weight regularization.
        activation (str, nn.Module, or callable): Activation function. Default: ``elu``.

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        num_heads: int,
        num_repeats: int,
        dropout: float = 0.1,
        conditional_channels: Optional[int] = None,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        self.num_repeats = num_repeats

        backbone = []

        for _ in range(num_repeats):
            block = ResidualBlock2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                causal=True,
                weight_regularization=weight_regularization,
                activation=activation,
                dropout=dropout,
                conditional_channels=conditional_channels,
                **factory_kwargs,
            )
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        self.q_proj = ResidualBlock2d(
            in_channels + 2,
            in_channels,
            kernel_size=1,
            causal=False,
            weight_regularization=weight_regularization,
            activation=activation,
            dropout=dropout,
            **factory_kwargs,
        )
        self.k_proj = ResidualBlock2d(
            2 * in_channels + 2,
            in_channels,
            kernel_size=1,
            causal=False,
            weight_regularization=weight_regularization,
            activation=activation,
            dropout=dropout,
            **factory_kwargs,
        )
        self.mha2d = CausalAttention2d(
            in_channels // 2,
            in_channels + 2,
            2 * in_channels + 2,
            num_heads=num_heads,
            dropout=dropout,
            weight_regularization=weight_regularization,
        )
        self.out_proj = ResidualBlock2d(
            in_channels,
            in_channels,
            kernel_size=1,
            weight_regularization=weight_regularization,
            auxiliary_channels=in_channels // 2,
            dropout=dropout,
            **factory_kwargs,
        )

    def forward(
        self,
        input: torch.Tensor,
        background: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of PixelBlock.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).
            background (torch.Tensor): Background embedding of shape
                (batch_size, 2, height, width) or (2, height, width).
            conditioning (torch.Tensor, optional): Conditional feature of
                shape (batch_size, conditional_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        num_repeats = self.num_repeats

        batch_size, _, height, width = input.size()

        if background.dim() == 3:
            x_background = background.unsqueeze(dim=0)
            x_background = x_background.expand(batch_size, 2, height, width)
        elif x_background.dim() == 4:
            x_background = background
        else:
            raise ValueError("background is expected 3D or 4D.")

        x = input

        for repeat_idx in range(num_repeats):
            x = self.backbone[repeat_idx](x, conditioning=conditioning)

        query = torch.cat([x, x_background], dim=1)
        query = self.q_proj(query)
        key = torch.cat([x, input, x_background], dim=1)
        key = self.k_proj(key)
        attn_output = self.mha2d(query, key)
        output = self.out_proj(x, attn_output)

        return output

    def weight_norm_(self) -> None:
        """Set weight_norm to modules."""
        for block in self.backbone:
            block: ResidualBlock2d
            block.weight_norm_()

        self.q_proj.weight_norm_()
        self.k_proj.weight_norm_()
        self.mha2d.weight_norm_()
        self.out_proj.weight_norm_()

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from module."""
        for block in self.backbone:
            block: ResidualBlock2d
            block.weight_norm_()

        self.q_proj.weight_norm_()
        self.k_proj.weight_norm_()
        self.mha2d.weight_norm_()
        self.out_proj.weight_norm_()


class ResidualBlock2d(nn.Module):
    """ResidualBlock for 2D input.

    Args:
        in_channels (int): Number of input channels.
        hidden_channels (int): Number of hidden channels.
        kernel_size (_size_2_t): Kernel size in convolutions.
        causal: Causality. If ``causal=True``, causal convolution is used.
        weight_regularization (str, optional): Weight regularization.
        activation (str, nn.Module, or callable): Activation function. Default: ``elu``.
        dropout (float): Dropout rate in attention. Default: ``0.0``.

    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t,
        causal: bool = False,
        weight_regularization: Optional[str] = "weight_norm",
        activation: Optional[
            Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
        ] = "elu",
        dropout: float = 0.1,
        auxiliary_channels: Optional[int] = None,
        conditional_channels: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        kernel_size = _pair(kernel_size)

        if causal:
            # To be compatible with definition in CausalConv2d,
            # re-define kernel_size.
            kh, kw = kernel_size
            kernel_size = 2 * kh - 1, kw

        self.kernel_size = kernel_size
        self.causal = causal

        if isinstance(activation, str):
            activation_1 = _get_activation(activation)
            activation_2 = _get_activation(activation)

            if auxiliary_channels is None:
                aux_activation = None
            else:
                aux_activation = _get_activation(activation)
        else:
            # NOTE: Activations are not shared with each other.
            activation_1 = copy.deepcopy(activation)
            activation_2 = copy.deepcopy(activation)

            if auxiliary_channels is None:
                aux_activation = None
            else:
                aux_activation = copy.deepcopy(activation)

        self.activation_1 = activation_1

        if causal:
            # NOTE: capture_center=True also ensures causality.
            self.conv2d_1 = PixelCNNCausalConv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                capture_center=False,
                **factory_kwargs,
            )
        else:
            self.conv2d_1 = nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=kernel_size,
                **factory_kwargs,
            )

        # auxiliary input
        if auxiliary_channels is None:
            self.aux_activation = None
            self.aux_conv2d = None
        else:
            self.aux_activation = aux_activation
            self.aux_conv2d = nn.Conv2d(
                auxiliary_channels,
                hidden_channels,
                kernel_size=1,
                **factory_kwargs,
            )

        self.activation_2 = activation_2
        self.dropout2d = nn.Dropout(dropout)

        if causal:
            # NOTE: capture_center=True also ensures causality.
            self.conv2d_2 = PixelCNNCausalConv2d(
                hidden_channels,
                2 * in_channels,
                kernel_size=kernel_size,
                capture_center=False,
                **factory_kwargs,
            )
        else:
            self.conv2d_2 = nn.Conv2d(
                hidden_channels,
                2 * in_channels,
                kernel_size=kernel_size,
                **factory_kwargs,
            )

        # conditional input
        if conditional_channels is None:
            self.conditional_conv2d = None
        else:
            self.conditional_conv2d = nn.Conv2d(
                conditional_channels,
                2 * in_channels,
                kernel_size=1,
                bias=False,
                **factory_kwargs,
            )

        self.glu = nn.GLU(dim=1)

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(
        self,
        input: torch.Tensor,
        auxiliary: Optional[torch.Tensor] = None,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).
            auxiliary (torch.Tensor, optional): Auxiliary feature of
                shape (batch_size, auxiliary_channels, height, width).
            conditioning (torch.Tensor, optional): Conditional feature of
                shape (batch_size, conditional_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, in_channels, height, width).

        """
        kh, kw = self.kernel_size

        padding_height = (kh - 1) // 2
        padding_width = (kw - 1) // 2
        padding_size = (padding_width, padding_width, padding_height, padding_height)

        x = self.activation_1(input)
        x = F.pad(x, padding_size)
        x = self.conv2d_1(x)

        if auxiliary is not None:
            x_auxiliary = self.aux_activation(auxiliary)
            x = x + self.aux_conv2d(x_auxiliary)

        x = self.activation_2(x)
        x = self.dropout2d(x)
        x = F.pad(x, padding_size)
        x = self.conv2d_2(x)

        if conditioning is not None:
            x = x + self.conditional_conv2d(conditioning)

        x = self.glu(x)
        output = x + input

        return output

    def weight_norm_(self) -> None:
        """Set weight_norm to convolutions."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.conv2d_1 = weight_norm_fn(self.conv2d_1)
        self.conv2d_2 = weight_norm_fn(self.conv2d_2)

        if self.aux_conv2d is not None:
            self.aux_conv2d = weight_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = weight_norm_fn(self.conditional_conv2d)

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from convolutions."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.conv2d_1 = remove_weight_norm_fn(self.conv2d_1, *remove_weight_norm_args)
        self.conv2d_2 = remove_weight_norm_fn(self.conv2d_2, *remove_weight_norm_args)

        if self.aux_conv2d is not None:
            self.aux_conv2d = remove_weight_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = remove_weight_norm_fn(self.conditional_conv2d)

    def spectral_norm_(self) -> None:
        """Set spectral_norm to convolutions."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.conv2d_1 = spectral_norm_fn(self.conv2d_1)
        self.conv2d_2 = spectral_norm_fn(self.conv2d_2)

        if self.aux_conv2d is not None:
            self.aux_conv2d = spectral_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = spectral_norm_fn(self.conditional_conv2d)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from convolutions."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.conv2d_1 = remove_spectral_norm_fn(self.conv2d_1, *remove_spectral_norm_args)
        self.conv2d_2 = remove_spectral_norm_fn(self.conv2d_2, *remove_spectral_norm_args)

        if self.aux_conv2d is not None:
            self.aux_conv2d = remove_spectral_norm_fn(self.aux_conv2d)

        if self.conditional_conv2d is not None:
            self.conditional_conv2d = remove_spectral_norm_fn(self.conditional_conv2d)


class CausalAttention2d(nn.Module):
    """Cross attention with causality for 2D input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Embedding dimension of values, which is equal to number of
            output channels. ``out_channels`` should be divisible by ``num_heads``.
        kdim (int): Embedding dimension of keys. ``kdim`` should be divisible by ``num_heads``.
        num_heads (int): Number of heads in attention.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        weight_regularization (str, optional): Weight regularization.

    """

    def __init__(
        self,
        embed_dim: int,
        qdim: int,
        kdim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.q_proj = nn.Linear(qdim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(kdim, embed_dim)

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.qdim, self.kdim = qdim, kdim
        self.num_heads = num_heads
        self.dropout = dropout

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Forward pass of CausalSelfAttention2d.

        Args:
            query (torch.Tensor): Query of shape (batch_size, qdim, height, width).
            key (torch.Tensor): Key of shape (batch_size, kdim, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, embed_dim, height, width).

        """
        embed_dim = self.embed_dim
        qdim, kdim = self.qdim, self.kdim
        num_heads = self.num_heads
        dropout = self.dropout
        batch_size, _, height, width = query.size()

        query = query.permute(0, 2, 3, 1).contiguous()
        query = query.view(batch_size, height * width, qdim)
        key = key.permute(0, 2, 3, 1).contiguous()
        key = key.view(batch_size, height * width, kdim)

        query = self.q_proj(query)
        value = self.v_proj(key)
        key = self.k_proj(key)
        query = query.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        key = key.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        value = value.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attn_score = torch.matmul(query, key) / math.sqrt(embed_dim // num_heads)
        attn_mask = self.generate_square_subsequent_mask(
            height * width,
            device=attn_score.device,
            dtype=attn_score.dtype,
        )
        attn_score = attn_score + attn_mask
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
        x = torch.matmul(attn_weights, value)
        x = x.permute(0, 1, 3, 2).contiguous()
        output = x.view(batch_size, embed_dim, height, width)

        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> torch.BoolTensor:
        return _generate_square_subsequent_mask(sz, device=device, dtype=dtype)

    def weight_norm_(self) -> None:
        """Set weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.q_proj = weight_norm_fn(self.q_proj)
        self.k_proj = weight_norm_fn(self.k_proj)
        self.v_proj = weight_norm_fn(self.v_proj)

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.q_proj = remove_weight_norm_fn(self.q_proj, *remove_weight_norm_args)
        self.k_proj = remove_weight_norm_fn(self.k_proj, *remove_weight_norm_args)
        self.v_proj = remove_weight_norm_fn(self.v_proj, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        """Set spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.q_proj = spectral_norm_fn(self.q_proj)
        self.k_proj = spectral_norm_fn(self.k_proj)
        self.v_proj = spectral_norm_fn(self.v_proj)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.q_proj = remove_spectral_norm_fn(self.q_proj, *remove_spectral_norm_args)
        self.k_proj = remove_spectral_norm_fn(self.k_proj, *remove_spectral_norm_args)
        self.v_proj = remove_spectral_norm_fn(self.v_proj, *remove_spectral_norm_args)
