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


class GumbelMNIST(MNIST):
    """MNIST dataset for GumbelVQVAE."""

    def __init__(
        self,
        root: str,
        init_temperature: float = 1,
        min_temperature: float | None = None,
        gamma: float = 1,
        last_epoch: int = -1,
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

        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.gamma = gamma
        self.epoch_index = last_epoch

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        init_temperature = self.init_temperature
        min_temperature = self.min_temperature
        gamma = self.gamma
        epoch_index = self.epoch_index

        if epoch_index < 0:
            raise ValueError("Call set_epoch before iteration.")

        data = super().__getitem__(index)

        temperature = init_temperature * (gamma**epoch_index)

        if min_temperature is not None:
            temperature = max(temperature, min_temperature)

        data["temperature"] = temperature

        return data

    def set_epoch(self, epoch: int) -> None:
        self.epoch_index = epoch
