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
        win_length: Optional[List[int]] = None,
        hop_length: Optional[List[int]] = None,
        weights: Optional[List[int]] = None,
        transform: Optional[Union[nn.ModuleList, nn.Sequential, bool]] = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.weights = weights
        self.eps = eps

        if win_length is None:
            self.win_length = []

            for idx in range(len(self.n_fft)):
                _n_fft = self.n_fft[idx]
                _win_length = _n_fft
                self.win_length.append(_win_length)
        else:
            assert len(self.win_length) == len(self.n_fft)

        if hop_length is None:
            self.hop_length = []

            for idx in range(len(self.n_fft)):
                _win_length = self.win_length[idx]
                _hop_length = _win_length // 4
                self.hop_length.append(_hop_length)
        else:
            assert len(self.hop_length) == len(self.n_fft)

        if weights is None:
            self.weights = []

            for idx in range(len(self.n_fft)):
                _hop_length = self.hop_length[idx]
                self.weights.append(math.sqrt(_hop_length / 2))
        else:
            assert len(self.weights) == len(self.n_fft)

        if type(transform) is bool:
            if not transform:
                raise ValueError(
                    "Set transform=True or specify transformation module as transform."
                )

            transform_modules = []

            for idx in range(len(self.n_fft)):
                _n_fft = self.n_fft[idx]
                _win_length = self.win_length[idx]
                _hop_length = self.hop_length[idx]

                _transform = aT.Spectrogram(_n_fft, win_length=_win_length, hop_length=_hop_length)
                transform_modules.append(_transform)

            transform = nn.ModuleList(transform_modules)
        elif isinstance(transform, (nn.ModuleList, nn.Sequential)):
            pass
        else:
            raise ValueError("Use nn.ModuleList or nn.Sequential as custom transform.")

        self.transform = transform

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of MultiScaleSpectralLoss.

        Args:
            input (torch.Tensor): Input waveform of shape (batch_size, length)
                or (batch_size, in_channels, length).
            target (torch.Tensor): Target waveform of shape (batch_size, length)
                or (batch_size, in_channels, length).

        .. note::

            Loss is averaged over batch, frequency, and time axes.

        """
        eps = self.eps
        loss = 0

        for idx in range(len(self.transform)):
            transform = self.transform[idx]
            weight = self.weights[idx]

            _input = transform(input)
            _target = transform(target)

            l1 = torch.abs(_target - _input)
            # average over frequency bin
            l1 = torch.mean(l1, dim=-2)
            # average over time frames and batch samples
            l1 = l1.mean()

            l2 = torch.log(_target + eps) - torch.log(_input + eps)
            # average over frequency bin
            l2 = torch.linalg.vector_norm(l2, ord=2, dim=-2) / l2.size(-2)
            # average over time frames and batch samples
            l2 = l2.mean()

            l12 = l1 + weight * l2
            loss = loss + l12

        return loss
