import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.modules.utils import _pair

from ..functional.activation import scaled_dot_product_attention
from .transformer import get_activation
from .vit import _PatchEmbedding

__all__ = [
    "MusicTaggingTransformerEncoder",
    "PositionalPatchEmbedding",
]

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")
IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class MusicTaggingTransformerEncoder(nn.Module):
    """Encoder of Music Tagging Transformer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        norm: Optional[nn.Module] = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        layers = []

        for _ in range(num_layers):
            layer = MusicTaggingTransformerEncoderLayer(
                d_model,
                nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                bias=bias,
                **factory_kwargs,
            )
            layers.append(layer)

        self.layers = nn.ModuleList(layers)
        self.norm = norm

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        is_causal: Optional[bool] = None,
    ) -> torch.Tensor:
        x = src

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        for layer in self.layers:
            x = layer(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        if self.norm is None:
            output = x
        else:
            output = self.norm(x)

        return output


class MusicTaggingTransformerEncoderLayer(nn.Module):
    """Encoder layer of Music Tagging Transformer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        if IS_TORCH_LT_2_1:
            assert bias, "Only bias=True is supported for torch < 2.1."

            layer_norm_kwargs = {}
        else:
            layer_norm_kwargs = {"bias": bias}

        self.self_attn = MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (torch.Tensor): the sequence to the encoder layer.
            src_mask (torch.BoolTensor, optional): the mask for the src sequence.
            src_key_padding_mask (torch.BoolTensor, optional): the mask for the src keys
                per batch.
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)


class MultiheadAttention(nn.Module):
    """Multihead attention for Music Tagging Transformer.

    For details, see ``torch.nn.MultiheadAttention``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty((embed_dim, embed_dim), **factory_kwargs)
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.kdim), **factory_kwargs)
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty((embed_dim, self.vdim), **factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        # in_proj_bias is always None in Music Tagging Transformer.
        self.register_parameter("in_proj_bias", None)

        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self._reset_parameters()

    def validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate keyword arguments for backward compatibility."""
        valid_keys = set()

        if not IS_TORCH_LT_2_0:
            valid_keys.add("is_causal")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."

    def _reset_parameters(self) -> None:
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)
        else:
            nn.init.xavier_uniform_(self.q_proj_weight)
            nn.init.xavier_uniform_(self.k_proj_weight)
            nn.init.xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)

        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of AbsolutePositionalMultiheadAttention.

        Args:
            query (torch.Tensor): Sequence of shape (batch_size, query_length, embed_dim)
                if ``batch_first=True``, otherwise (query_length, batch_size, embed_dim).
            key (torch.Tensor): Sequence of shape (batch_size, key_length, embed_dim)
                if ``batch_first=True``, otherwise (key_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, key_length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (query_length, key_length) or
                (batch_size * num_heads, query_length, key_length).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Sequence of same shape as input.
                - torch.Tensor: Attention weights of shape
                    (batch_size, num_heads, query_length, key_length) if
                    ``average_attn_weights=True``, otherwise
                    (batch_size, query_length, key_length).

        """
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias

        head_dim = embed_dim // num_heads

        if batch_first:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query = query.transpose(1, 0)
                    key = key.transpose(1, 0)
                    value = key
            else:
                query = query.transpose(1, 0)
                key = key.transpose(1, 0)
                value = value.transpose(1, 0)

        query_length, batch_size, _ = query.size()
        key_length, _, _ = key.size()

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if self._qkv_same_embed_dim:
            q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
                in_proj_weight, [embed_dim] * 3, dim=-2
            )
        else:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight

        if self.in_proj_bias is None:
            q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            # NOTE: in_proj_bias is always None in Music Tagging Transformer.
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = F.linear(query, q_proj_weight, bias=q_proj_bias)
        k = F.linear(key, k_proj_weight, bias=k_proj_bias)
        v = F.linear(value, v_proj_weight, bias=v_proj_bias)

        q = q.view(query_length, batch_size, num_heads, head_dim)
        k = k.view(key_length, batch_size, num_heads, head_dim)
        v = v.view(key_length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, query_length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(query_length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights and need_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights


class PositionalPatchEmbedding(_PatchEmbedding):
    """Patch embedding + trainable positional embeddings for Music Tagging Transformer."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_channels: int,
        n_bins: int,
        kernel_size: _size_2_t = 3,
        pool_kernel_size: Union[_size_2_t, List[_size_2_t]] = None,
        pool_stride: Optional[Union[_size_2_t, List[_size_2_t]]] = None,
        num_layers: int = 3,
        num_blocks: int = 2,
        insert_cls_token: bool = True,
        insert_dist_token: bool = False,
        dropout: float = 0,
        max_length: int = 512,
        support_extrapolation: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__(
            embedding_dim,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            **factory_kwargs,
        )

        self.max_length = max_length
        self.support_extrapolation = support_extrapolation

        if pool_kernel_size is None and pool_stride is None:
            pool_kernel_size = [(2, 2)] * num_layers
            pool_stride = [(2, 2)] * (num_layers - 1)
            pool_stride.append((2, 1))

        if isinstance(pool_kernel_size, list):
            pass
        else:
            pool_kernel_size = [pool_kernel_size] * num_layers

        if pool_stride is None:
            pool_stride = [None] * num_layers

        if isinstance(pool_stride, list):
            pass
        else:
            pool_stride = [pool_stride] * num_layers

        self.batch_norm = nn.BatchNorm2d(1)

        num_features = n_bins
        backbone = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                _in_channels = 1
            else:
                _in_channels = hidden_channels

            _out_channels = hidden_channels
            _pool_kernel_size = pool_kernel_size[layer_idx]
            _pool_stride = pool_stride[layer_idx]

            layer = ResidualMaxPool2d(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                pool_kernel_size=_pool_kernel_size,
                pool_stride=_pool_stride,
                num_blocks=num_blocks,
            )
            backbone.append(layer)
            (_pool_stride, _) = _pair(layer.pool2d.stride)
            num_features //= _pool_stride

        self.backbone = nn.ModuleList(backbone)
        self.fc = nn.Linear(num_features * hidden_channels, embedding_dim)

        positional_embedding = torch.empty((embedding_dim, max_length), **factory_kwargs)
        self.positional_embedding = nn.Parameter(positional_embedding)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        self.reset_head_tokens()

        # based on official implementation
        nn.init.normal_(self.positional_embedding.data)

    def reset_head_tokens(self) -> None:
        # based on official implementation
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token.data)

        if self.dist_token is not None:
            nn.init.normal_(self.dist_token.data)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResidualFrontend2d.

        Args:
            input (torch.Tensor): Spectrogram-like feature of shape (batch_size, n_bins, n_frames)

        Returns:
            torch.Tensor: Downsampled feature of shape (batch_size, n_frames', out_channels).

        """
        positional_embedding = self.positional_embedding

        x = input.unsqueeze(dim=-3)
        x = self.batch_norm(x)

        for layer in self.backbone:
            x = layer(x)

        batch_size, _, _, length = x.size()
        max_length = positional_embedding.size(-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, length, -1)
        x = self.fc(x)

        if length > max_length and not self.support_extrapolation:
            warnings.warn(
                "Number of time frames is greater than predefined value.",
                UserWarning,
                stacklevel=2,
            )
            x, _ = torch.split(x, [max_length, length - max_length], dim=-1)
        else:
            positional_embedding = self.resample_positional_embedding(
                positional_embedding,
                length,
            )

        x = x + positional_embedding.permute(1, 0)

        if self.insert_dist_token:
            dist_token = self.dist_token.expand((batch_size, 1, -1))
            x = torch.cat([dist_token, x], dim=-2)

        if self.insert_cls_token:
            cls_token = self.cls_token.expand((batch_size, 1, -1))
            x = torch.cat([cls_token, x], dim=-2)

        output = self.dropout(x)

        return output

    def resample_positional_embedding(
        self,
        positional_embedding: Union[torch.Tensor],
        length: int,
        mode: str = "bilinear",
    ) -> torch.Tensor:
        """Resample positional embedding.

        Args:
            positional_embedding (torch.Tensor): Positional embedding of shape
                (embedding_dim, max_length).
            length (int): Target length.
            mode (str): Interpolation mode. Default: ``bilinear``.

        Returns:
            torch.Tensor: Resampled positional embedding of shape (embedding_dim, length).

        """
        _, max_length = positional_embedding.size()

        if max_length > length:
            output, _ = torch.split(
                positional_embedding,
                [length, max_length - length],
                dim=-1,
            )
        elif length > max_length:
            positional_embedding = positional_embedding.view(1, -1, 1, max_length)
            positional_embedding = F.interpolate(positional_embedding, size=(1, length), mode=mode)
            output = positional_embedding.view(-1, length)
        else:
            output = positional_embedding

        return output

    def compute_output_shape(self, n_bins: int, n_frames: int) -> int:
        max_length = self.positional_embedding.size(-1)

        height, width = n_bins, n_frames

        for layer in self.backbone:
            layer: ResidualMaxPool2d
            height, width = layer.compute_output_shape(height, width)

        length = width

        if length > max_length and not self.support_extrapolation:
            warnings.warn(
                "Number of time frames is greater than predefined value.",
                UserWarning,
                stacklevel=2,
            )
            length = max_length

        return length


class ResidualMaxPool2d(nn.Module):
    """Max pooling block used for PositionalPatchEmbedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        pool_kernel_size: _size_2_t = 2,
        pool_stride: Optional[_size_2_t] = None,
        num_blocks: int = 2,
    ) -> None:
        super().__init__()

        stride = 1
        backbone = []

        for block_idx in range(num_blocks):
            if block_idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = out_channels

            block = ConvBlock2d(_in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            backbone.append(block)

        self.backbone = nn.ModuleList(backbone)

        if in_channels == out_channels:
            self.post_block2d = None
        else:
            self.post_block2d = ConvBlock2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride
            )

        self.relu2d = nn.ReLU()
        self.pool2d = nn.MaxPool2d(pool_kernel_size, stride=pool_stride)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ResidualMaxPool2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height', width').

        """
        x = input

        for block in self.backbone:
            x = block(x)

        if self.post_block2d is not None:
            residual = self.post_block2d(input)
            x = x + residual

        x = self.relu2d(x)
        output = self.pool2d(x)

        return output

    def compute_output_shape(self, height: int, width: int) -> Tuple[int, int]:
        # Stride is assumed to be 1 in convolution blocks.
        # Pooling changes output shape.
        (kernel_height, kernel_width) = _pair(self.pool2d.kernel_size)
        (stride_height, stride_width) = _pair(self.pool2d.stride)
        height = (height - kernel_height) // stride_height + 1
        width = (width - kernel_width) // stride_width + 1

        return height, width


class ConvBlock2d(nn.Module):
    """Convolution + batch normalization block for Music Tagging Transformer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 1,
    ) -> None:
        super().__init__()

        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.batch_norm2d = nn.BatchNorm2d(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of ConvBlock2d.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, out_channels, height', width').

        """
        Kh, Kw = _pair(self.conv2d.kernel_size)
        Sh, Sw = _pair(self.conv2d.stride)
        Ph = Kh - Sh
        Pw = Kw - Sw
        padding_top = Ph // 2
        padding_bottom = Ph - padding_top
        padding_left = Pw // 2
        padding_right = Pw - padding_left

        x = F.pad(input, (padding_left, padding_right, padding_top, padding_bottom))
        x = self.conv2d(x)
        output = self.batch_norm2d(x)

        return output
