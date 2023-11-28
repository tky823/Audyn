from typing import Union

import torch.nn as nn

from ..data import select_device
from ..parallel import is_dp_or_ddp

__all__ = ["set_device"]


def set_device(
    module: nn.Module, accelerator: str, is_distributed: bool = False
) -> Union[nn.Module, nn.parallel.DistributedDataParallel]:
    device = select_device(accelerator, is_distributed=is_distributed)
    module = module.to(device)

    trainable = any(p.requires_grad for p in module.parameters())

    if is_distributed and trainable:
        module = nn.parallel.DistributedDataParallel(module, device_ids=[device])

    return module


def unwrap(module: nn.Module) -> nn.Module:
    if is_dp_or_ddp(module):
        module: Union[nn.parallel.DataParallel, nn.parallel.DistributedDataParallel]
        return module.module
    else:
        return module
