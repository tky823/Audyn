from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseVAE(ABC, nn.Module):
    @abstractmethod
    def encode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement encode method.")

    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Implement decode method.")
