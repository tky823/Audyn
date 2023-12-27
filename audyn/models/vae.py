from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn


class BaseVAE(ABC, nn.Module):
    @abstractmethod
    def encode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement encode method.")

    @abstractmethod
    def decode(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Implement decode method.")

    @abstractmethod
    def sample(self, *args, **kwargs) -> Any:
        """Non-differentiable sampling."""
        raise NotImplementedError("Implement sample method.")

    @abstractmethod
    def rsample(self, *args, **kwargs) -> Any:
        """Differentiable sampling with reparametrization trick."""
        raise NotImplementedError("Implement rsample method.")
