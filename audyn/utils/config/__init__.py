from dataclasses import dataclass
from typing import TypeVar

from omegaconf import DictConfig

SystemConfig = TypeVar("SystemConfig", "_SystemConfig", DictConfig)


@dataclass
class _SystemConfig:
    seed: int
    distributed: "DistributedConfig"
    cudnn: "CUDNNConfig"
    amp: "AMPConfig"
    accelerator: str


@dataclass
class DistributedConfig:
    enable: bool
    backend: str
    init_method: str


@dataclass
class CUDNNConfig:
    benchmark: bool
    deterministic: bool


@dataclass
class AMPConfig:
    enable: bool


@dataclass
class TrainConfig:
    system: SystemConfig
