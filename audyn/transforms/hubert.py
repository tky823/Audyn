from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as aT

from .kaldi import KaldiMFCC

__all__ = ["HuBERTMFCC"]


class HuBERTMFCC(nn.Module):
    """MFCC + delta + delta-delta features for HuBERT.

    Args:
        sample_rate (int): Sampling rate.
        mfcc_kwargs (dict, optional): Keyword arguments given to KaldiMFCC.
        deltas_kwargs (dict, optional): Keyword arguments given to
            torchaudio.transforms.ComputeDeltas.

    """

    def __init__(
        self,
        sample_rate: int,
        mfcc_kwargs: Optional[Dict[str, Any]] = None,
        deltas_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if mfcc_kwargs is None:
            mfcc_kwargs = {
                "use_energy": False,
            }

        if deltas_kwargs is None:
            deltas_kwargs = {}

        self.mfcc_transform = KaldiMFCC(sample_rate, mfcc_kwargs=mfcc_kwargs)
        self.deltas_transform = aT.ComputeDeltas(**deltas_kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Extract HuBERT features.

        Args:
            input (torch.Tensor): Waveform of shape (timesteps,).

        Returns:
            torch.Tensor: Extracted features of shape (n_bins, n_frames),
                where n_bins is 3 * n_mfcc.

        """
        mfcc = self.mfcc_transform(input)
        deltas = self.deltas_transform(mfcc)
        ddeltas = self.deltas_transform(deltas)

        output = torch.cat([mfcc, deltas, ddeltas], dim=0)

        return output
