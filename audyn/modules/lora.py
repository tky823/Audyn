import copy
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from ..functional.activation import scaled_dot_product_attention

__all__ = [
    "LoRALinear",
]

IS_TORCH_LT_2_0 = version.parse(torch.__version__) < version.parse("2.0")


class LoRALinear(nn.Module):
    """Linear layer for low-rank adaptation.

    Args:
        weight (nn.Parameter or torch.Tensor, optional): Weight parameter in ``nn.Linear``.
        bias (nn.Parameter or torch.Tensor, optional): Bias parameter in ``nn.Linear``.
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

    """

    def __init__(
        self,
        weight: Union[nn.Parameter, torch.Tensor],
        bias: Optional[Union[nn.Parameter, torch.Tensor]] = None,
        rank: int = 8,
        persistent: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

        super().__init__()

        weight = copy.copy(weight.data)

        if bias is not None:
            bias = copy.copy(bias.data)

        # register weight and bias as buffer
        self.register_buffer("weight", weight, persistent=persistent)
        self.register_buffer("bias", bias, persistent=persistent)

        out_features, in_features = weight.size()

        self.weight_in = nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))
        self.weight_out = nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = F.linear(input, self.weight, bias=self.bias)
        x_lora = F.linear(input, self.weight_in)
        x_lora = F.linear(x_lora, self.weight_out)
        output = x + x_lora

        return output

    def _reset_parameters(self) -> None:
        std = 1 / math.sqrt(self.rank)
        self.weight_in.data.normal_(std=std)
        self.weight_out.data.zero_()

    def extra_repr(self) -> str:
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        s += f", bias={self.bias is not None}"
        s += f", rank={self.rank}"

        return s

    @classmethod
    def build_from_linear(
        cls,
        module: nn.Linear,
        rank: int = 8,
        persistent: bool = False,
    ) -> "LoRALinear":
        weight = module.weight
        bias = module.bias

        factory_kwargs = {
            "dtype": weight.dtype,
            "device": weight.device,
        }

        module = cls(
            weight,
            bias=bias,
            rank=rank,
            persistent=persistent,
            **factory_kwargs,
        )

        return module


class LoRAMultiheadAttention(nn.Module):
    """Multihead attention for low-rank adaptation.

    Args:
        rank (int): Rank of weight matrices. Small value (e.g. 8) is expected in LoRA.
        persistent (bool): If ``persistent=True``, original ``weight`` and ``bias`` are
            stored in ``state_dict``. Default: ``False``.

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
        rank: int = 8,
        persistent: bool = False,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> None:
        factory_kwargs = {
            "dtype": dtype,
            "device": device,
        }

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

        q_proj_weight = copy.deepcopy(q_proj_weight.data)
        k_proj_weight = copy.deepcopy(k_proj_weight.data)
        v_proj_weight = copy.deepcopy(v_proj_weight.data)

        if _qkv_same_embed_dim:
            in_proj_weight = torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=-2)
            self.register_buffer("in_proj_weight", in_proj_weight, persistent=persistent)
            self.register_buffer("q_proj_weight", None, persistent=persistent)
            self.register_buffer("k_proj_weight", None, persistent=persistent)
            self.register_buffer("v_proj_weight", None, persistent=persistent)

            self.in_proj_weight_in = nn.Parameter(
                torch.empty((3 * rank, embed_dim), **factory_kwargs)
            )
        else:
            self.register_buffer("in_proj_weight", None, persistent=persistent)
            self.register_buffer("q_proj_weight", q_proj_weight, persistent=persistent)
            self.register_buffer("k_proj_weight", k_proj_weight, persistent=persistent)
            self.register_buffer("v_proj_weight", v_proj_weight, persistent=persistent)

            self.q_proj_weight_in = nn.Parameter(torch.empty((rank, embed_dim), **factory_kwargs))
            self.k_proj_weight_in = nn.Parameter(torch.empty((rank, _kdim), **factory_kwargs))
            self.v_proj_weight_in = nn.Parameter(torch.empty((rank, _vdim), **factory_kwargs))

        self.q_proj_weight_out = nn.Parameter(torch.empty((embed_dim, rank), **factory_kwargs))
        self.k_proj_weight_out = nn.Parameter(torch.empty((embed_dim, rank), **factory_kwargs))
        self.v_proj_weight_out = nn.Parameter(torch.empty((embed_dim, rank), **factory_kwargs))

        if in_proj_bias is None:
            self.register_buffer("in_proj_bias", None, persistent=persistent)
        else:
            in_proj_bias = copy.deepcopy(in_proj_bias.data)
            self.register_buffer("in_proj_bias", in_proj_bias, persistent=persistent)

        self.out_proj = LoRALinear(
            out_proj_weight,
            bias=out_proj_bias,
            rank=rank,
            persistent=persistent,
            **factory_kwargs,
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

        self.rank = rank

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        std = 1 / math.sqrt(self.rank)

        if self.in_proj_weight_in is None:
            self.q_proj_weight_in.data.normal_(std=std)
            self.k_proj_weight_in.data.normal_(std=std)
            self.v_proj_weight_in.data.normal_(std=std)
        else:
            self.in_proj_weight_in.data.normal_(std=std)

        self.q_proj_weight_out.data.zero_()
        self.k_proj_weight_out.data.zero_()
        self.v_proj_weight_out.data.zero_()

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
        in_proj_weight_in = self.in_proj_weight_in
        q_proj_weight_out = self.q_proj_weight_out
        k_proj_weight_out = self.k_proj_weight_out
        v_proj_weight_out = self.v_proj_weight_out
        rank = self.rank

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
            q_proj_weight_in, k_proj_weight_in, v_proj_weight_in = torch.split(
                in_proj_weight_in, [rank] * 3, dim=-2
            )
        else:
            q_proj_weight = self.q_proj_weight
            k_proj_weight = self.k_proj_weight
            v_proj_weight = self.v_proj_weight
            q_proj_weight_in = self.q_proj_weight_in
            k_proj_weight_in = self.k_proj_weight_in
            v_proj_weight_in = self.v_proj_weight_in

        if self.in_proj_bias is None:
            q_proj_bias, k_proj_bias, v_proj_bias = None, None, None
        else:
            q_proj_bias, k_proj_bias, v_proj_bias = torch.split(
                in_proj_bias, [embed_dim] * 3, dim=0
            )

        q = F.linear(query, q_proj_weight, bias=q_proj_bias)
        k = F.linear(key, k_proj_weight, bias=k_proj_bias)
        v = F.linear(value, v_proj_weight, bias=v_proj_bias)

        q_lora = F.linear(query, q_proj_weight_in)
        q_lora = F.linear(q_lora, q_proj_weight_out)
        q = q + q_lora
        k_lora = F.linear(key, k_proj_weight_in)
        k_lora = F.linear(k_lora, k_proj_weight_out)
        k = k + k_lora
        v_lora = F.linear(value, v_proj_weight_in)
        v_lora = F.linear(v_lora, v_proj_weight_out)
        v = v + v_lora

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
