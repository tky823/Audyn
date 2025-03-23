from typing import Callable, Dict, Optional

import torch
from torchvision.datasets import CIFAR100 as BaseCIFAR100


class CIFAR100(BaseCIFAR100):
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


class GumbelCIFAR100(CIFAR100):
    """CIFAR100 dataset for GumbelVQVAE.

    Args:
        schedule (str): Schedule of temperature. ``linear`` and ``exponential``
            are supported.
    """

    def __init__(
        self,
        root: str,
        init_temperature: float = 1,
        min_temperature: float | None = None,
        schedule: str = "linear",
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

        assert schedule in ["linear", "exponential"], f"Invalid schedule {schedule} is given."
        assert gamma > 0, "gamma should be positive."

        self.init_temperature = init_temperature
        self.min_temperature = min_temperature
        self.schedule = schedule
        self.gamma = gamma
        self._step = last_epoch

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        init_temperature = self.init_temperature
        min_temperature = self.min_temperature
        schedule = self.schedule
        gamma = self.gamma
        step = self._step

        if step < 0:
            raise ValueError("Call set_step before iteration.")

        data = super().__getitem__(index)

        if schedule == "linear":
            temperature = init_temperature - step * gamma
        elif schedule == "exponential":
            temperature = init_temperature * (gamma**step)
        else:
            raise ValueError(f"Invalid schedule {schedule} is given.")

        if min_temperature is not None:
            temperature = max(temperature, min_temperature)

        data["temperature"] = temperature

        return data

    def set_step(self, step: int) -> None:
        self._step = step

    def get_step(self) -> int:
        return self._step
