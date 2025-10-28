import copy
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

from ..functional.activation import scaled_dot_product_attention
from .activation import RotaryPositionalMultiheadAttention
from .glu import GLU1d
from .positional_encoding import RotaryPositionalEmbedding
from .transformer import get_activation

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class Conv2dSubsampling(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: _size_2_t = 3,
        stride: Union[int, List[int]] = 2,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_bins: int = 128,
    ) -> None:
        super().__init__()

        if isinstance(stride, int):
            stride = [stride] * 2
        else:
            assert len(stride) == 2, "Stride must be int or list of two ints."

        backbone = []

        for layer_index in range(num_layers):
            if layer_index == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels

            _out_channels = hidden_channels
            _stride = (2, stride[layer_index])

            block = ResidualBlock2d(_in_channels, _out_channels, kernel_size, stride=_stride)
            backbone.append(block)

        self.backbone = nn.Sequential(*backbone)
        self.proj = nn.Linear(hidden_channels * n_bins // (2**num_layers), out_channels)
        self.dropout = nn.Dropout(p=dropout)

        self.n_bins = n_bins

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of Conv2dSubsampling.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames).

        Returns:
            torch.Tensor: Subsampled feature of shape (batch_size, n_frames', out_channels).

        """
        n_bins = self.n_bins

        assert input.size(-2) == n_bins, (
            f"Shape of input should be (*, n_bins={n_bins}, n_frames)."
        )

        x = self.backbone(input)
        batch_size, _n_channels, _n_bins, _n_frames = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(batch_size, _n_frames, _n_channels * _n_bins)
        x = self.proj(x)
        output = self.dropout(x)

        return output


class ConformerEncoderLayer(nn.Module):
    """Conformer encoder layer."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        feedforward_activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
        conv_activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.silu,
        layer_norm_eps: float = 1e-5,
        rope_base: int = 10000,
        share_heads: bool = True,
        rope_format: str = "conformer",
        rope_dtype: Optional[torch.dtype] = None,
        batch_first: bool = False,
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

        self.self_attn = ConformerRotaryPositionalMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            base=rope_base,
            share_heads=share_heads,
            batch_first=batch_first,
            rope_format=rope_format,
            rope_dtype=rope_dtype,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(p=dropout)

        self.glu = GLU1d(d_model, d_model, kernel_size=1, bias=False)
        self.conv1d1 = nn.Conv1d(
            d_model, d_model, kernel_size=31, stride=1, padding=15, groups=d_model, bias=False
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.conv1d2 = nn.Conv1d(d_model, d_model, kernel_size=1, stride=1, bias=False)

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.dropout2 = nn.Dropout(p=dropout)

        self.linear3 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout3 = nn.Dropout(p=dropout)
        self.linear4 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.dropout4 = nn.Dropout(p=dropout)

        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm4 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm5 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )

        if isinstance(feedforward_activation, str):
            feedforward_activation = get_activation(feedforward_activation)

        if isinstance(conv_activation, str):
            conv_activation = get_activation(conv_activation)

        self.feedforward_activation = copy.deepcopy(feedforward_activation)
        self.conv_activation = copy.deepcopy(conv_activation)

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
        x = x + 0.5 * self._ff_block1(self.norm1(x))
        x = x + self._sa_block(self.norm2(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._conv_block(self.norm3(x))
        x = x + 0.5 * self._ff_block2(self.norm4(x))

        return self.norm5(x)

    @property
    def batch_first(self) -> bool:
        return self.self_attn.batch_first

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
            attn_mask=attn_mask,
            padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout(x)

    def _conv_block(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Convolution module.

        Args:
            x (torch.Tensor): Input feature of shape (batch_size, length, d_model).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, length, d_model).

        """
        if self.batch_first:
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(1, 2, 0)

        x = self.glu(x)
        x = self.conv1d1(x)
        x = self.norm(x)
        x = self.conv_activation(x)
        x = self.conv1d2(x)
        x = self.dropout(x)

        if self.batch_first:
            output = x.permute(0, 2, 1)
        else:
            output = x.permute(2, 0, 1)

        return output

    def _ff_block1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.feedforward_activation(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        return self.dropout2(x)

    def _ff_block2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear3(x)
        x = self.feedforward_activation(x)
        x = self.dropout3(x)
        x = self.linear4(x)

        return self.dropout4(x)


class ResidualBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t = 3,
        stride: _size_2_t = 2,
    ) -> None:
        super().__init__()

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        Kh, Kw = kernel_size
        Sh, Sw = stride
        padding = Kh - Sh, Kw - Sw

        self.conv2d1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            stride=stride,
        )
        self.norm2d1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.norm2d2 = nn.BatchNorm2d(out_channels)

        if out_channels != in_channels or stride != (1, 1):
            self.conv2d3 = nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding, stride=stride
            )
            self.norm2d3 = nn.BatchNorm2d(out_channels)
        else:
            self.conv2d3 = nn.Identity()
            self.norm2d3 = nn.Identity()

        self.relu2 = nn.ReLU()

    def forward(self, input: torch.Tensor):
        x = self.conv2d1(input)
        x = self.norm2d1(x)
        x = self.relu1(x)
        x = self.conv2d2(x)
        x = self.norm2d2(x)

        x_residual = self.conv2d3(input)
        x_residual = self.norm2d3(x_residual)

        x = x + x_residual
        output = self.relu2(x)

        return output
        return output


class ConformerRotaryPositionalMultiheadAttention(RotaryPositionalMultiheadAttention):
    """Multihead attention using rotary positional representation for Conformer."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        base: int = 10000,
        share_heads: bool = True,
        rope_format: str = "conformer",
        rope_dtype: Optional[torch.dtype] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super(RotaryPositionalMultiheadAttention, self).__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self.rope = ConformerRotaryPositionalEmbedding(
            base=base,
            batch_first=batch_first,
            rope_format=rope_format,
            rope_dtype=rope_dtype,
        )

        self.share_heads = share_heads

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of RotaryPositionalMultiheadAttention.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, embed_dim)
                if ``batch_first=True``, otherwise (length, batch_size, embed_dim).
            padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, length).
            attn_mask (torch.BoolTensor, optional): Attention padding mask of
                shape (length, length) or (batch_size * num_heads, length, length).

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
            x = input.transpose(1, 0)
        else:
            x = input

        length, batch_size, _ = x.size()

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
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        tgt = x
        src = self._apply_positional_embedding(x.contiguous())

        q = F.linear(src, q_proj_weight, bias=q_proj_bias)
        k = F.linear(src, k_proj_weight, bias=k_proj_bias)
        v = F.linear(tgt, v_proj_weight, bias=v_proj_bias)

        q = q.view(length, batch_size, num_heads, head_dim)
        k = k.view(length, batch_size, num_heads, head_dim)
        v = v.view(length, batch_size, num_heads, head_dim)

        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        dropout_p = dropout if self.training else 0

        qkv, attn_weights = scaled_dot_product_attention(
            q,
            k,
            v,
            key_padding_mask=padding_mask,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            need_weights=need_weights,
        )

        if batch_first:
            qkv = qkv.permute(0, 2, 1, 3).contiguous()
            qkv = qkv.view(batch_size, length, embed_dim)
        else:
            qkv = qkv.permute(2, 0, 1, 3).contiguous()
            qkv = qkv.view(length, batch_size, embed_dim)

        output = self.out_proj(qkv)

        if average_attn_weights and need_weights:
            attn_weights = attn_weights.mean(dim=1)

        if not need_weights:
            attn_weights = None

        return output, attn_weights


class ConformerRotaryPositionalEmbedding(RotaryPositionalEmbedding):
    """RoPE: Rotary positional embedding for Conformer.

    Args:
        rope_format (str): Format of RoPE. Choose from ``"conformer"`` or ``"roformer"``.
        rope_dtype (Optional[torch.dtype]): Data type of RoPE. If ``rope_format=="conformer"``,
            ``torch.bfloat16`` is used by default for compatibility with official Conformer
            implementation.

    """

    def __init__(
        self,
        base: int = 10000,
        batch_first: bool = True,
        rope_format: str = "conformer",
        rope_dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__(
            base=base,
            batch_first=batch_first,
        )

        if rope_format == "conformer" and rope_dtype is None:
            rope_dtype = torch.bfloat16

        self.rope_format = rope_format
        self.rope_dtype = rope_dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of RoPE for Conformer.

        Args:
            input (torch.Tensor): Sequence of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Sequence of same shape as input.

        """
        base = self.base
        batch_first = self.batch_first
        rope_dtype = self.rope_dtype

        dtype = input.dtype
        device = input.device

        if batch_first:
            x_cos = input
        else:
            x_cos = input.transpose(0, 1)

        batch_size, length, num_features = x_cos.size()

        x_pre, x_post = x_cos.chunk(chunks=2, dim=-1)
        x_sin = torch.cat([-x_post, x_pre], dim=-1)
        x_sin = x_sin.view(batch_size, length, 2, num_features // 2)
        x_cos = x_cos.view(batch_size, length, 2, num_features // 2)

        pos_seq = torch.arange(length)
        num_seq = torch.arange(0, num_features, 2) / num_features

        if self.rope_format == "conformer":
            scale = 1 / (base**num_seq)

            if rope_dtype is not None:
                scale = scale.to(rope_dtype)

            theta = pos_seq.unsqueeze(-1) * scale.to(dtype)
        elif self.rope_format == "roformer":
            theta = pos_seq.unsqueeze(-1) / (base**num_seq)
        else:
            raise ValueError(f"Unknown rope_format: {self.rope_format}")

        sin = torch.sin(theta)
        cos = torch.cos(theta)
        sin = sin.to(device)
        cos = cos.to(device)

        x = x_cos * cos.unsqueeze(dim=-2) + x_sin * sin.unsqueeze(dim=-2)
        x = x.view(batch_size, length, num_features)

        if batch_first:
            output = x
        else:
            output = x.transpose(0, 1).contiguous()

        return output
