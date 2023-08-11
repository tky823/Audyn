from typing import Union

import torch.nn as nn

from ..data import select_device

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
