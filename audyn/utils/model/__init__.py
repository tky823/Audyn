import warnings
from typing import Any, Dict, Union

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from ...metrics.base import StatefulMetric
from ..modules import set_device as _set_device
from ..modules import unwrap as _unwrap

__all__ = [
    "set_device",
    "unwrap",
]


def set_device(
    module: Union[nn.Module, StatefulMetric],
    accelerator: str,
    is_distributed: bool = False,
    ddp_kwargs: Dict[str, Any] = None,
) -> Union[nn.Module, DistributedDataParallel]:
    warnings.warn(
        "Use audyn.utils.modules.set_device instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    return _set_device(
        module,
        accelerator,
        is_distributed=is_distributed,
        ddp_kwargs=ddp_kwargs,
    )


def unwrap(module: nn.Module) -> nn.Module:
    warnings.warn(
        "Use audyn.utils.modules.unwrap instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _unwrap(module)
