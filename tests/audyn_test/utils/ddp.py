import functools
import os
from typing import Any, Callable

import torch

from audyn_test.utils import reset_random_port


def retry_on_file_not_found(num_retries: int, enable: bool = True) -> Callable:
    """Retry function or method up to num_retries times when FileNotFoundError is detected.

    Args:
        num_retries (int): Maximum retry.
        enable (bool): Whether to activate or deactivate.

    """

    def wrapper(func: Any) -> Callable:
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            attempts = 0

            if enable:
                while attempts <= num_retries:
                    try:
                        return func(*args, **kwargs)
                    except FileNotFoundError as e:
                        attempts += 1

                        if attempts > num_retries:
                            raise e

                        reset_random_port()
            else:
                return func(*args, **kwargs)

        return _wrapper

    return wrapper


def set_ddp_environment(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)
