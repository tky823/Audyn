import math
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torchaudio.transforms as aT

__all__ = ["MultiScaleSpectralLoss"]


class MultiScaleSpectralLoss(nn.Module):
    """Multi-scale spectral reconstruction loss."""

    def __init__(
        self,
        n_fft: List[int],
        hop_length: Optional[List[int]] = None,
        weights: Optional[List[int]] = None,
        transform: Optional[Union[nn.Module, bool]] = True,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.weights = weights

        if type(transform) is bool:
            if not transform:
                raise ValueError(
                    "Set transform=True or specify transformation module as transform."
                )

            transform_modules = []

            if weights is None:
                self.weights = []

            for idx in range(len(self.n_fft)):
                n_fft = self.n_fft[idx]

                if self.hop_length is None:
                    hop_length = n_fft // 4
                else:
                    hop_length = self.hop_length[idx]

                if weights is None:
                    self.weights.append(math.sqrt(hop_length / 2))

                _transform = aT.Spectrogram(n_fft, hop_length=hop_length)
                transform_modules.append(_transform)

            transform = nn.ModuleList(transform_modules)

        self.transform = transform

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiScaleSpectralLoss.

        Args:
            input (torch.Tensor): Input waveform of shape (batch_size, length)
                or (batch_size, in_channels, length).
            target (torch.Tensor): Target waveform of shape (batch_size, length)
                or (batch_size, in_channels, length).

        """
        loss = 0

        for idx in range(len(self.transform)):
            transform = self.transform[idx]
            weight = self.weights[idx]

            _input = transform(input)
            _target = transform(target)

            l1 = torch.abs(_target - _input)
            l1 = torch.sum(l1, dim=-2)
            l1 = l1.sum()

            l2 = torch.log(_target) - torch.log(_input)
            l2 = torch.linalg.vector_norm(l2, ord=2, dim=-2)
            l2 = l2.sum()

            l12 = l1 + weight * l2
            loss = loss + l12

        return loss
