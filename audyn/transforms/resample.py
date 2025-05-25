from typing import Tuple

import torch
import torch.nn as nn
import torchaudio.functional as aF

__all__ = [
    "DynamicResample",
]


class DynamicResample(nn.Module):
    """Module of torchaudio.functional.resample.

    Unlike ``torchaudio.transforms.Resample``, ``DynamicResample`` allows dynamic resampling.
    """

    def __init__(self, sample_rate: int, **kwargs) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.kwargs = kwargs

    def forward(self, waveform: torch.Tensor, sample_rate: int) -> Tuple[torch.Tensor, int]:
        kwargs = self.kwargs

        if sample_rate != self.sample_rate:
            waveform = aF.resample(waveform, sample_rate, self.sample_rate, **kwargs)

        return waveform
