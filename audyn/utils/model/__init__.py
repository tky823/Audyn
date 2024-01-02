from typing import Any, Dict, Union

import torch.nn as nn

from ..data import select_device
from ..parallel import is_dp_or_ddp

__all__ = ["set_device"]


def set_device(
    module: nn.Module,
    accelerator: str,
    is_distributed: bool = False,
    ddp_kwargs: Dict[str, Any] = None,
) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
    """Set device of module.

    Args:
        module (nn.Module): Module to set device.
        accelerator (str): Accelerator of module. ``cpu``, ``gpu``, ``cuda``, and ``mps``
            are supported.
        is_distributed (bool): Whether to use ``nn.parallel.DistributedDataParallel``.
        ddp_kwargs (dict, optional): Keyword arguments given to
            ``nn.parallel.DistributedDataParallel``. e.g. ``find_unused_parameters``.

    Returns:
        nn.Module: Module allocated on specified device.

    """
    device = select_device(accelerator, is_distributed=is_distributed)
    module = module.to(device)

    trainable = any(p.requires_grad for p in module.parameters())

    if is_distributed and trainable:
        if ddp_kwargs is None:
            ddp_kwargs = {}

        module = nn.parallel.DistributedDataParallel(module, device_ids=[device], **ddp_kwargs)

    return module


def unwrap(module: nn.Module) -> nn.Module:
    if is_dp_or_ddp(module):
        module: Union[nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]
        return module.module
    else:
        return module
