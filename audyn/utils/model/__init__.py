from typing import Any, Dict, Union

import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from ...criterion.base import MultiCriteria
from ...metrics.base import StatefulMetric
from ..data import select_device
from ..parallel import is_dp_or_ddp

__all__ = ["set_device"]


def set_device(
    module: Union[nn.Module, StatefulMetric],
    accelerator: str,
    is_distributed: bool = False,
    ddp_kwargs: Dict[str, Any] = None,
) -> Union[nn.Module, DistributedDataParallel]:
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

    if isinstance(module, nn.Module):
        trainable = any(p.requires_grad for p in module.parameters())

        if is_distributed and trainable:
            if ddp_kwargs is None:
                ddp_kwargs = {}

            if isinstance(module, MultiCriteria):
                for criterion_name in module.keys():
                    criterion = module[criterion_name]
                    trainable = any(p.requires_grad for p in criterion.parameters())

                    if trainable:
                        module[criterion_name] = DistributedDataParallel(
                            criterion, device_ids=[device], **ddp_kwargs
                        )
            else:
                module = DistributedDataParallel(module, device_ids=[device], **ddp_kwargs)
    elif isinstance(module, StatefulMetric):
        pass
    else:
        raise ValueError(f"{type(module)} is not supported.")

    return module


def unwrap(module: nn.Module) -> nn.Module:
    if is_dp_or_ddp(module):
        module: Union[DataParallel, DistributedDataParallel]
        return module.module
    else:
        return module
