import functools
from typing import Any, Callable


def run_only_master_rank(enable: bool = True) -> Callable:
    def wrapper(m: Any) -> Callable:
        @functools.wraps(m)
        def _wrapper(mod: Any, *args, **kwargs) -> Any:
            if hasattr(mod, "is_distributed"):
                is_distributed = mod.is_distributed
            else:
                is_distributed = False

            if hasattr(mod, "global_rank"):
                global_rank = mod.global_rank
            else:
                global_rank = 0

            if global_rank is None:
                global_rank = 0

            if enable and is_distributed and global_rank != 0:
                return

            return m(mod, *args, **kwargs)

        return _wrapper

    return wrapper
