import os

import torch
import torch.distributed as dist
from omegaconf import DictConfig

__all__ = ["setup_distributed", "is_distributed"]


def setup_distributed(config: DictConfig) -> None:
    """Set up distributed system of torch.

    Args:
        config (DictConfig): Config to set up distributed system.

    .. note::

        The following configuration is required at least:

        ```
        distributed:
            enable: True  # should be True.
            backend:  # If None, nccl is used by default.
            init_method:  # optional

        ```

    """
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    if config.distributed.backend is None:
        backend = "nccl"
    else:
        backend = config.distributed.backend

    dist.init_process_group(
        backend=backend,
        init_method=config.distributed.init_method,
        rank=global_rank,
        world_size=world_size,
    )


def is_distributed(config: DictConfig) -> bool:
    """Examine availability of distributed system.

    Args:
        config (DictConfig): Config of system.

    .. note::

        The following configuration is required at least:

        ```
        distributed:
            enable: true # true, false or none (optional)

        ```

    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        availability = str(config.distributed.enable).lower()

        if num_gpus > 1:
            if availability == "false":
                raise ValueError(
                    "Set config.system.distributed.enable=true for multi GPU training."
                )
            else:
                is_distributed = True
        else:
            if availability == "true":
                is_distributed = True
            else:
                is_distributed = False
    else:
        is_distributed = False

    return is_distributed
