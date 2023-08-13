from typing import Any

import torch.nn as nn

__all__ = ["is_dp_or_ddp"]


def is_dp_or_ddp(module: Any) -> bool:
    """Judge whether model is a subclass of nn.parallel.DataParallel
    or nn.parallel.DistributedDataParallel.

    Args:
        module (any): Certain module.

    Returns:
        bool: If model is a subclass of nn.parallel.DataParallel
            or nn.parallel.DistributedDataParallel, ``True`` is returned.
            Otherwise, ``False``.

    """
    if isinstance(module, nn.parallel.DataParallel):
        return True
    elif isinstance(module, nn.parallel.DistributedDataParallel):
        return True
    else:
        return False
