import copy
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn

from ...modules.bitnet import (
    BitLinear158,
    BitLinear158Inference,
    BitMultiheadAttention158,
    BitMultiheadAttention158Inference,
)

__all__ = [
    "convert_to_bitlinear158",
    "convert_to_bitlinear158_inference",
    "convert_linear_to_bitlinear158",
    "convert_mha_to_bitmha158",
    "convert_bitlinear158_to_bitlinear158_inference",
    "convert_bitmha158_to_bitmha158_inference",
]


def convert_to_bitlinear158(
    module: nn.Module,
    norm: Optional[Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
    remove_bias: bool = False,
) -> nn.Module:
    """Convert nn.Linear and nn.MultiheadAttention to BitLinear158 and BitMultiheadAttention158,
    respectively in given module.

    Args:
        module (nn.Module): Module to be converted.
        bits (int): Number of quantization bits in scale parameter.
        eps (float): Tiny value to avoid zero division.
        remove_bias (bool): If ``True``, bias is forced to be removed.

    Returns:
        nn.Module: Converted module.

    """
    if isinstance(module, (BitLinear158, BitMultiheadAttention158)):
        module = copy.deepcopy(module)
    elif isinstance(module, nn.Linear):
        module = convert_linear_to_bitlinear158(
            module,
            norm=copy.deepcopy(norm),
            dim=dim,
            bits=bits,
            eps=eps,
            remove_bias=remove_bias,
        )
    elif isinstance(module, nn.MultiheadAttention):
        module = convert_mha_to_bitmha158(
            module,
            norm=copy.deepcopy(norm),
            dim=dim,
            bits=bits,
            eps=eps,
            remove_bias=remove_bias,
        )
    else:
        for name, child_module in module.named_children():
            child_module = convert_to_bitlinear158(
                child_module,
                norm=copy.deepcopy(norm),
                dim=dim,
                bits=bits,
                eps=eps,
                remove_bias=remove_bias,
            )

            setattr(module, name, child_module)

    return module


def convert_to_bitlinear158_inference(
    module: nn.Module,
    norm: Optional[Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
    remove_bias: bool = False,
) -> nn.Module:
    """Convert nn.Linear and nn.MultiheadAttention to BitLinear158Inference and
    BitMultiheadAttention158Inference, respectively in given module.

    BitLinear158 and BitMultiheadAttention158 are also converted to their
    inference modules.

    Args:
        module (nn.Module): Module to be converted.
        bits (int): Number of quantization bits in scale parameter.
        eps (float): Tiny value to avoid zero division.
        remove_bias (bool): If ``True``, bias is forced to be removed.

    Returns:
        nn.Module: Converted module.

    """
    module = convert_to_bitlinear158(
        module,
        norm=copy.deepcopy(norm),
        dim=dim,
        bits=bits,
        eps=eps,
        remove_bias=remove_bias,
    )

    if isinstance(module, (BitLinear158Inference, BitMultiheadAttention158Inference)):
        module = copy.deepcopy(module)
    elif isinstance(module, BitLinear158):
        module = convert_bitlinear158_to_bitlinear158_inference(module)
    elif isinstance(module, BitMultiheadAttention158):
        module = convert_bitmha158_to_bitmha158_inference(module)
    else:
        for name, child_module in module.named_children():
            child_module = convert_to_bitlinear158_inference(
                child_module,
                dim=dim,
                bits=bits,
                eps=eps,
                remove_bias=remove_bias,
            )

            setattr(module, name, child_module)

    return module


def convert_linear_to_bitlinear158(
    module: nn.Linear,
    norm: Optional[Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
    remove_bias: bool = False,
) -> BitLinear158:
    """Convert nn.Linear to BitLinear158."""
    factory_kwargs = {
        "device": module.weight.device,
        "dtype": module.weight.dtype,
    }

    if module.bias is None or remove_bias:
        bias = False
    else:
        bias = True

    in_features = module.in_features
    out_features = module.out_features

    converted = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        norm=copy.deepcopy(norm),
        dim=dim,
        bits=bits,
        eps=eps,
        **factory_kwargs,
    )
    converted.weight.data.copy_(module.weight.data)

    if bias:
        converted.bias.data.copy_(module.bias.data)

    return converted


def convert_mha_to_bitmha158(
    module: nn.MultiheadAttention,
    norm: Optional[Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None,
    dim: Optional[Union[int, Sequence[int]]] = None,
    bits: int = 8,
    eps: float = 1e-5,
    remove_bias: bool = False,
) -> BitMultiheadAttention158:
    """Convert nn.MultiheadAttention to BitMultiheadAttention158.

    .. note::

        Even when ``remove_bias=True``, ``bias_k`` and ``bias_v``
        are not removed from BitMultiheadAttention158. The role of these values
        is not the bias term in nn.Linear.

    """
    # device and dtype is determined by out_proj
    factory_kwargs = {
        "device": module.out_proj.weight.device,
        "dtype": module.out_proj.weight.dtype,
    }

    embed_dim = module.embed_dim
    num_heads = module.num_heads
    dropout = module.dropout
    bias = module.in_proj_bias is not None
    add_bias_kv = module.bias_k is not None
    add_zero_attn = module.add_zero_attn
    kdim = module.kdim
    vdim = module.vdim
    batch_first = module.batch_first

    if bias:
        assert module.out_proj.bias is not None

        if remove_bias:
            bias = False
    else:
        assert module.out_proj.bias is None

    if add_bias_kv:
        assert module.bias_v is not None
    else:
        assert module.bias_v is None

    converted = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        dropout=dropout,
        bias=bias,
        add_bias_kv=add_bias_kv,
        add_zero_attn=add_zero_attn,
        kdim=kdim,
        vdim=vdim,
        batch_first=batch_first,
        norm=copy.deepcopy(norm),
        dim=dim,
        bits=bits,
        eps=eps,
        **factory_kwargs,
    )

    if module.in_proj_weight is None:
        converted.q_proj_weight.data.copy_(module.q_proj_weight.data)
        converted.k_proj_weight.data.copy_(module.k_proj_weight.data)
        converted.v_proj_weight.data.copy_(module.v_proj_weight.data)
    else:
        converted.in_proj_weight.data.copy_(module.in_proj_weight.data)

    if module.in_proj_bias is not None and bias:
        converted.in_proj_bias.data.copy_(module.in_proj_bias.data)

    converted.out_proj.weight.data.copy_(module.out_proj.weight.data)

    if module.out_proj.bias is not None and bias:
        converted.out_proj.bias.data.copy_(module.out_proj.bias.data)

    if module.bias_k is not None:
        converted.bias_k.data.copy_(module.bias_k.data)

    if module.bias_v is not None:
        converted.bias_v.data.copy_(module.bias_v.data)

    return converted


def convert_bitlinear158_to_bitlinear158_inference(module: BitLinear158) -> BitLinear158Inference:
    """Convert BitLinear158 to BitLinear158Inference."""
    converted = BitLinear158Inference.build_from_bitlinear158(module)

    return converted


def convert_bitmha158_to_bitmha158_inference(
    module: BitMultiheadAttention158,
) -> BitMultiheadAttention158Inference:
    """Convert BitMultiheadAttention158 to BitMultiheadAttention158Inference."""
    converted = BitMultiheadAttention158Inference.build_from_bitmha158(module)

    return converted
