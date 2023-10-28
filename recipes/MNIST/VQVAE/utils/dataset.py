from typing import Callable, Dict, Optional

import torch
from torchvision.datasets import MNIST as BaseMNIST


class MNIST(BaseMNIST):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        input, target = super().__getitem__(index)

        data = {
            "input": input,
            "target": target,
            "index": index,
        }

        return data
