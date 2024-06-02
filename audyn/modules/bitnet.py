import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional.activation import scaled_dot_product_attention
from ..functional.bitnet import bitlinear158, bitlinear158_inference, quantize_weight
from .activation import _MultiheadAttention

__all__ = [
    "BitLinear158",
    "BitMultiheadAttention158",
    "BitLinear158Inference",
    "RoundClip",
]


class BitLinear158(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 8,
        bias: bool = True,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        weight = torch.empty(
            (out_features, in_features),
            **factory_kwargs,
        )
        weight = nn.Parameter(weight, requires_grad=True)
        self.register_parameter("weight", weight)

        if bias:
            bias = torch.empty(
                (out_features,),
                **factory_kwargs,
            )
            bias = nn.Parameter(bias, requires_grad=True)
            self.register_parameter("bias", bias)
        else:
            self.register_parameter("bias", None)

        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.eps = eps

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = bitlinear158(
            input,
            self.weight,
            bias=self.bias,
            bits=self.bits,
            eps=self.eps,
        )

        return output

    def _reset_parameters(self) -> None:
        # ported from https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py#L105-L113  # noqa: E501
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        in_features = self.in_features
        out_features = self.out_features
        bits = self.bits
        bias = self.bias is not None

        return f"in_features={in_features}, out_features={out_features}, bits={bits}, bias={bias}"


class BitMultiheadAttention158(_MultiheadAttention):
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
        bits: int = 8,
        eps: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **kwargs,
        )

        from ..utils.model.bitnet import convert_linear_to_bitlinear158

        self.out_proj = convert_linear_to_bitlinear158(
            self.out_proj,
            bits=bits,
            eps=eps,
            remove_bias=False,
        )

        self.bits = bits
        self.eps = eps

        if add_bias_kv:
            raise NotImplementedError("add_bias_kv is not supported.")

        if add_zero_attn:
            raise NotImplementedError("add_zero_attn is not supported.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        embed_dim = self.embed_dim
        dropout = self.dropout
        batch_first = self.batch_first
        num_heads = self.num_heads
        in_proj_weight = self.in_proj_weight
        in_proj_bias = self.in_proj_bias
        bits = self.bits
        eps = self.eps

        head_dim = embed_dim // num_heads

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
            bits=bits,
            eps=eps,
        )
        k = bitlinear158(
            key,
            k_proj_weight,
            bias=k_proj_bias,
            bits=bits,
            eps=eps,
        )
        v = bitlinear158(
            value,
            v_proj_weight,
            bias=v_proj_bias,
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


class BitLinear158Inference(BitLinear158):
    """BitLinear158 for inference.

    Unlike ``BitLinear158``, quantization is performed during initialization.
    """

    def __init__(
        self,
        weight: Union[nn.Parameter, torch.Tensor],
        bits: int = 8,
        bias: Optional[Union[nn.Parameter, torch.Tensor]] = None,
        eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        weight = weight.data

        if bias is not None:
            bias = bias.data

        out_features, in_features = weight.size()

        super().__init__(
            in_features,
            out_features,
            bits=bits,
            bias=bias is not None,
            eps=eps,
            **factory_kwargs,
        )

        quantized_weight, beta = quantize_weight(weight, eps=self.eps)

        # remove weight parameter set quantized weight and beta buffers instead
        del self._parameters["weight"]
        self.register_buffer("quantized_weight", quantized_weight)
        self.register_buffer("beta", beta)

        if bias is not None:
            # remove bias parameter set bias buffer instead
            del self._parameters["bias"]
            self.register_buffer("bias", bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = bitlinear158_inference(
            input,
            self.quantized_weight,
            self.beta,
            bias=self.bias,
            bits=self.bits,
            eps=self.eps,
        )

        return output

    @classmethod
    def build_from_bitlinear158(cls, module: BitLinear158) -> "BitLinear158Inference":
        weight = module.weight.data

        if module.bias is None:
            bias = None
        else:
            bias = module.bias.data

        factory_kwargs = {
            "device": weight.device,
            "dtype": weight.dtype,
        }

        converted = cls(
            weight,
            bits=module.bits,
            bias=bias,
            eps=module.eps,
            **factory_kwargs,
        )

        return converted


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
