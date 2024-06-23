import functools
import os
import warnings
from typing import Any, Callable

import torch.distributed as dist

__all__ = [
    "run_only_global_master_rank",
    "run_only_local_master_rank",
    "run_only_master_rank",
]


def run_only_global_master_rank(enable: bool = True) -> Callable:
    def wrapper(m: Any) -> Callable:
        @functools.wraps(m)
        def _wrapper(mod: Any, *args, **kwargs) -> Any:
            if hasattr(mod, "is_distributed"):
                is_distributed = mod.is_distributed
            else:
                is_distributed = dist.is_available() and dist.is_initialized()

            if hasattr(mod, "global_rank"):
                global_rank = mod.global_rank
            else:
                if is_distributed:
                    global_rank = int(os.environ["RANK"])
                else:
                    global_rank = 0

            if enable and is_distributed and global_rank != 0:
                return

            return m(mod, *args, **kwargs)

        return _wrapper

    return wrapper


def run_only_local_master_rank(enable: bool = True) -> Callable:
    def wrapper(m: Any) -> Callable:
        @functools.wraps(m)
        def _wrapper(mod: Any, *args, **kwargs) -> Any:
            if hasattr(mod, "is_distributed"):
                is_distributed = mod.is_distributed
            else:
                is_distributed = dist.is_available() and dist.is_initialized()

            if hasattr(mod, "local_rank"):
                local_rank = mod.local_rank
            else:
                if is_distributed:
                    local_rank = int(os.environ["LOCAL_RANK"])
                else:
                    local_rank = 0

            if enable and is_distributed and local_rank != 0:
                return

            return m(mod, *args, **kwargs)

        return _wrapper

    return wrapper


def run_only_master_rank(enable: bool = True) -> Callable:
    warnings.warn(
        "run_only_master_rank is deprecated. Use run_only_global_master_rank instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_only_global_master_rank(enable)
