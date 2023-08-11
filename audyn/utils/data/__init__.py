import os
from typing import Optional

from torch.utils.data import DataLoader


class BaseDataLoaders:
    def __init__(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
    ) -> None:
        self.train = train_loader
        self.validation = validation_loader


def select_device(accelerator: str, is_distributed: bool = False) -> str:
    if accelerator in ["cuda", "gpu"] and is_distributed:
        device = int(os.environ["LOCAL_RANK"])
    elif accelerator in ["cpu", "cuda", "mps"]:
        device = accelerator
    elif accelerator == "gpu":
        device = "cuda"
    else:
        raise ValueError(f"Accelerator {accelerator} is not supported.")

    return device
