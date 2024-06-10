import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear

from ..functional.activation import scaled_dot_product_attention
from ..functional.bitnet import bitlinear158, bitlinear158_inference, quantize_weight

__all__ = [
    "BitLinear158",
    "BitMultiheadAttention158",
    "BitLinear158Inference",
    "BitMultiheadAttention158Inference",
    "RoundClip",
]

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")


class BitLinear158(nn.Module):
    """BitLinear using ternary (i.e. 1.58bit) weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dim: Optional[Union[int, Sequence[int]]] = None,
        bits: int = 8,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        weight = torch.empty((out_features, in_features), **factory_kwargs)
        weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter("weight", weight)

        if bias:
            bias = torch.empty((out_features,), **factory_kwargs)
            bias = nn.Parameter(bias, requires_grad=True)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features

        self.dim = dim
        self.bits = bits
        self.eps = eps

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = bitlinear158(
            input,
            self.weight,
            bias=self.bias,
            dim=self.dim,
            bits=self.bits,
            eps=self.eps,
        )

        return output

    def _reset_parameters(self) -> None:
        # https://github.com/pytorch/pytorch/blob/b66e3f0957b96b058c9b632ca60833d9717a9d8a/torch/nn/modules/linear.py#L106-L114
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        in_features = self.in_features
        out_features = self.out_features
        bias = self.bias is not None
        dim = self.dim
        bits = self.bits

        return (
            f"in_features={in_features}, out_features={out_features}, bias={bias}"
            f", dim={dim}, bits={bits}"
        )


class BitMultiheadAttention158(nn.Module):
    """Multihead attention using BitLinear158.

    For parameters, see details of nn.MultiheadAttention.
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
        dim: Optional[Union[int, Sequence[int]]] = None,
        bits: int = 8,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        from ..utils.model.bitnet import convert_linear_to_bitlinear158

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

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
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

        self.dim = dim
        self.bits = bits
        self.eps = eps

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

        self._reset_parameters()

        # convert out_proj after self._reset_parameters
        self.out_proj = convert_linear_to_bitlinear158(
            self.out_proj,
            dim=dim,
            bits=bits,
            eps=eps,
            remove_bias=False,
        )

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
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        bits = self.bits
        eps = self.eps

        head_dim = embed_dim // num_heads
        # dim is used for in_proj only, not used for out_proj
        dim = swap_group_dim_if_necessary(
            self.dim,
            query.size(),
            batch_first=batch_first,
        )

        if batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

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
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = bitlinear158(
            query,
            q_proj_weight,
            bias=q_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )
        k = bitlinear158(
            key,
            k_proj_weight,
            bias=k_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )
        v = bitlinear158(
            value,
            v_proj_weight,
            bias=v_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )

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
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)


class BitLinear158Inference(nn.Module):
    """BitLinear158 for inference.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    def __init__(
        self,
        weight: Union[nn.Parameter, torch.Tensor],
        dim: Optional[Union[int, Sequence[int]]] = None,
        bits: int = 8,
        bias: Optional[Union[nn.Parameter, torch.Tensor]] = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        quantized_weight, scale = quantize_weight(weight.data)

        if bias is not None:
            bias = bias.data

        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("scale", scale)

        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias)

        self.in_features, self.out_features = weight.size()
        self.dim = dim
        self.bits = bits
        self.eps = eps

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = bitlinear158_inference(
            input,
            self.quantized_weight,
            self.scale,
            bias=self.bias,
            dim=self.dim,
            bits=self.bits,
            eps=self.eps,
        )

        return output

    @classmethod
    def build_from_bitlinear158(cls, module: BitLinear158) -> "BitLinear158Inference":
        converted = cls(
            module.weight,
            dim=module.dim,
            bits=module.bits,
            bias=module.bias,
            eps=module.eps,
        )

        return converted

    def extra_repr(self) -> str:
        in_features = self.in_features
        out_features = self.out_features
        bias = self.bias is not None
        dim = self.dim
        bits = self.bits

        return (
            f"in_features={in_features}, out_features={out_features}, bias={bias}"
            f", dim={dim}, bits={bits}"
        )


class BitMultiheadAttention158Inference(nn.Module):
    """Multihead attention using BitLinear158 for inference.

    For parameters, see details of nn.MultiheadAttention.
    """

    def __init__(
        self,
        num_heads: int,
        in_proj_weight: torch.Tensor,
        in_proj_bias: Optional[torch.Tensor],
        bias_k: Optional[torch.Tensor] = None,
        bias_v: Optional[torch.Tensor] = None,
        add_zero_attn: bool = False,
        dropout: float = 0.0,
        out_proj_weight: torch.Tensor = None,
        out_proj_bias: Optional[torch.Tensor] = None,
        q_proj_weight: Optional[torch.Tensor] = None,
        k_proj_weight: Optional[torch.Tensor] = None,
        v_proj_weight: Optional[torch.Tensor] = None,
        batch_first: bool = False,
        dim: Optional[Union[int, Sequence[int]]] = None,
        bits: int = 8,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if bias_k is None:
            add_bias_kv = False
        else:
            add_bias_kv = True

        if add_bias_kv:
            assert bias_v is not None
        else:
            assert bias_v is None

        if out_proj_weight is None:
            raise ValueError("out_proj_weight is required.")

        if q_proj_weight is None:
            assert in_proj_weight is not None
            assert k_proj_weight is None
            assert v_proj_weight is None

            embed_dim = in_proj_weight.size(-1)
            kdim = vdim = None
            _kdim = _vdim = embed_dim
        else:
            assert in_proj_weight is None
            assert k_proj_weight is not None
            assert v_proj_weight is not None

            embed_dim = q_proj_weight.size(-1)
            _kdim = kdim = k_proj_weight.size(-1)
            _vdim = vdim = v_proj_weight.size(-1)

        if embed_dim == _kdim and embed_dim == _vdim:
            _qkv_same_embed_dim = True
        else:
            _qkv_same_embed_dim = False

        if _qkv_same_embed_dim:
            q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
                in_proj_weight, [embed_dim] * 3, dim=-2
            )

        quantized_q_proj_weight, q_proj_scale = quantize_weight(q_proj_weight, eps=eps)
        quantized_k_proj_weight, k_proj_scale = quantize_weight(k_proj_weight, eps=eps)
        quantized_v_proj_weight, v_proj_scale = quantize_weight(v_proj_weight, eps=eps)

        if _qkv_same_embed_dim:
            quantized_in_proj_weight = torch.cat(
                [quantized_q_proj_weight, quantized_k_proj_weight, quantized_v_proj_weight], dim=-2
            )
            in_proj_scale = torch.stack([q_proj_scale, k_proj_scale, v_proj_scale], dim=0)
            self.register_buffer("quantized_in_proj_weight", quantized_in_proj_weight)
            self.register_buffer("in_proj_scale", in_proj_scale)
            self.register_buffer("quantized_q_proj_weight", None)
            self.register_buffer("quantized_k_proj_weight", None)
            self.register_buffer("quantized_v_proj_weight", None)
            self.register_buffer("q_proj_scale", None)
            self.register_buffer("k_proj_scale", None)
            self.register_buffer("v_proj_scale", None)

        else:
            self.register_buffer("quantized_in_proj_weight", None)
            self.register_buffer("in_proj_scale", None)
            self.register_buffer("quantized_q_proj_weight", quantized_q_proj_weight)
            self.register_buffer("quantized_k_proj_weight", quantized_k_proj_weight)
            self.register_buffer("quantized_v_proj_weight", quantized_v_proj_weight)
            self.register_buffer("q_proj_scale", q_proj_scale)
            self.register_buffer("k_proj_scale", k_proj_scale)
            self.register_buffer("v_proj_scale", v_proj_scale)

        self.register_buffer("in_proj_bias", in_proj_bias)

        self.out_proj = BitLinear158Inference(
            out_proj_weight,
            dim=dim,
            bits=bits,
            bias=out_proj_bias,
            eps=eps,
        )

        self.kdim = kdim
        self.vdim = vdim
        self._qkv_same_embed_dim = _qkv_same_embed_dim

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.dropout = dropout
        self.batch_first = batch_first
        self.dim = dim
        self.bits = bits
        self.eps = eps

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
        self.validate_kwargs(kwargs)

        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        quantized_in_proj_weight = self.quantized_in_proj_weight
        in_proj_scale = self.in_proj_scale
        in_proj_bias = self.in_proj_bias
        bits = self.bits
        eps = self.eps

        head_dim = embed_dim // num_heads
        # dim is used for in_proj only, not used for out_proj
        dim = swap_group_dim_if_necessary(
            self.dim,
            query.size(),
            batch_first=batch_first,
        )

        if batch_first:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

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
            quantized_q_proj_weight, quantized_k_proj_weight, quantized_v_proj_weight = (
                torch.split(quantized_in_proj_weight, [embed_dim] * 3, dim=-2)
            )
            q_proj_scale, k_proj_scale, v_proj_scale = torch.unbind(in_proj_scale)
        else:
            quantized_q_proj_weight = self.quantized_q_proj_weight
            quantized_k_proj_weight = self.quantized_k_proj_weight
            quantized_v_proj_weight = self.quantized_v_proj_weight
            q_proj_scale = self.q_proj_scale
            k_proj_scale = self.k_proj_scale
            v_proj_scale = self.v_proj_scale

        if self.in_proj_bias is None:
            q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = bitlinear158_inference(
            query,
            quantized_q_proj_weight,
            q_proj_scale,
            bias=q_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )
        k = bitlinear158_inference(
            key,
            quantized_k_proj_weight,
            k_proj_scale,
            bias=k_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )
        v = bitlinear158_inference(
            value,
            quantized_v_proj_weight,
            v_proj_scale,
            bias=v_proj_bias,
            dim=dim,
            bits=bits,
            eps=eps,
        )

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

    @classmethod
    def build_from_bitmha158(
        cls, module: BitMultiheadAttention158
    ) -> "BitMultiheadAttention158Inference":
        converted = cls(
            module.num_heads,
            module.in_proj_weight,
            module.in_proj_bias,
            bias_k=module.bias_k,
            bias_v=module.bias_v,
            add_zero_attn=module.add_zero_attn,
            dropout=module.dropout,
            out_proj_weight=module.out_proj.weight,
            out_proj_bias=module.out_proj.bias,
            q_proj_weight=module.q_proj_weight,
            k_proj_weight=module.k_proj_weight,
            v_proj_weight=module.v_proj_weight,
            batch_first=module.batch_first,
            dim=module.dim,
            bits=module.bits,
            eps=module.eps,
        )

        return converted

    def validate_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate keyword arguments for backward compatibility."""
        valid_keys = set()

        if not IS_TORCH_LT_2_0:
            valid_keys.add("is_causal")

        invalid_keys = set(kwargs.keys()) - valid_keys

        assert invalid_keys == set(), f"Invalid keys {invalid_keys} are given."


class RoundClip(nn.Module):
    """Apply min-max clipping after rounding.

    .. note::

        Gradient backpropagation is realized by straight through estimator.

    """

    def __init__(self, min: float = -1, max: float = 1) -> None:
        super().__init__()

        self.min = min
        self.max = max

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = torch.round(input)
        x = torch.clamp(x, min=self.min, max=self.max)
        output = torch.detach(x - input) + input

        return output


def swap_group_dim_if_necessary(
    dim: Optional[Union[int, Sequence[int]]],
    size: torch.Size,
    batch_first: bool = False,
) -> Optional[Union[int, Sequence[int]]]:
    """Swap group dimension of batch and token axes."""
    if dim is None:
        return dim

    is_int = isinstance(dim, int)

    if is_int:
        dim = [dim]
        _type = None
    else:
        _type = type(dim)

    new_dim = []

    for _dim in dim:
        # set index to non-negative
        if _dim < 0:
            _dim = len(size) + _dim

        if batch_first:
            # swap 0 and 1
            if _dim in [0, 1]:
                _dim = 1 - _dim

        new_dim.append(_dim)

    if is_int:
        new_dim = new_dim[0]
    else:
        new_dim = _type(new_dim)

    return new_dim
