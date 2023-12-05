import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist
from omegaconf import DictConfig

__all__ = ["setup_distributed", "is_distributed", "select_local_rank", "select_global_rank"]


def setup_distributed(config: DictConfig) -> None:
    """Set up distributed system of torch.

    Args:
        config (DictConfig): Config to set up distributed system.

    .. note::

        The following configuration is required at least:

        ```
        distributed:
            enable: true  # should be True.
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
    availability = str(config.distributed.enable).lower()

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()

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
        if availability == "true":
            warnings.warn(
                "config.system.distributed.enable is set to true, but CUDA is NOT available."
            )

        is_distributed = False

    return is_distributed


def select_local_rank(accelerator: Optional[str], is_distributed: bool = False) -> Optional[int]:
    if accelerator is None:
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    if accelerator in ["cuda", "gpu"] and is_distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
    elif accelerator in ["cpu", "gpu", "cuda", "mps"]:
        local_rank = None
    else:
        raise ValueError(f"Accelerator {accelerator} is not supported.")

    return local_rank


def select_global_rank(accelerator: Optional[str], is_distributed: bool = False) -> Optional[int]:
    if accelerator is None:
        if torch.cuda.is_available():
            accelerator = "cuda"
        else:
            accelerator = "cpu"

    if accelerator in ["cuda", "gpu"] and is_distributed:
        global_rank = int(os.environ["RANK"])
    elif accelerator in ["cpu", "gpu", "cuda", "mps"]:
        global_rank = None
    else:
        raise ValueError(f"Accelerator {accelerator} is not supported.")

    return global_rank
