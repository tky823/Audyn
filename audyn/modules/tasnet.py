import copy
from typing import Any, Callable, Dict, Optional, Union

import torch
import torch.nn as nn

from .normalization import CumulativeLayerNorm1d, GlobalLayerNorm

__all__ = [
    "get_nonlinear",
    "get_layer_norm",
]


def get_nonlinear(
    nonlinear: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    nonlinear_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if nonlinear_kwargs is None:
        nonlinear_kwargs = {}

    if isinstance(nonlinear, nn.Module):
        nonlinear = copy.deepcopy(nonlinear)

    if callable(nonlinear):
        assert len(nonlinear_kwargs) == 0
        return nonlinear

    assert isinstance(nonlinear, str)

    if nonlinear == "sigmoid":
        assert len(nonlinear_kwargs) == 0

        nonlinear = nn.Sigmoid()
    elif nonlinear == "softmax":
        nonlinear = nn.Softmax(**nonlinear_kwargs)
    else:
        raise NotImplementedError(f"{nonlinear} is not supported as nonlinear function.")

    return nonlinear


def get_layer_norm(
    norm: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]],
    num_features: Optional[int] = None,
    eps: float = 1e-8,
    **kwargs,
) -> Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if isinstance(norm, nn.Module):
        norm = copy.deepcopy(norm)

    if callable(norm):
        return norm

    assert isinstance(norm, str)
    assert num_features is not None

    if norm == "cLN":
        norm = CumulativeLayerNorm1d(num_features, eps=eps)
    elif norm == "gLN":
        norm = GlobalLayerNorm(num_features, eps=eps)
    elif norm in ["BN", "batch", "batch_norm"]:
        n_dims = kwargs.get("n_dims") or 1

        if n_dims == 1:
            norm = nn.BatchNorm1d(num_features, eps=eps)
        elif n_dims == 2:
            norm = nn.BatchNorm2d(num_features, eps=eps)
        else:
            raise NotImplementedError(f"n_dims is expected 1 or 2, but {n_dims} is given.")
    else:
        raise NotImplementedError(f"{norm} is not supported as normalization.")

    return norm
