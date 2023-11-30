from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as aT

__all__ = ["MelSpectrogramL1Loss"]


class MelSpectrogramL1Loss(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: Optional[float] = 0,
        f_max: Optional[float] = None,
        pad: int = 0,
        n_mels: int = 128,
        window_fn: Callable[[Any], torch.Tensor] = torch.hann_window,
        power: float = 2,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: Optional[str] = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        take_log: bool = False,
        reduction: str = "mean",
        clamp_min: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.take_log = take_log
        self.clamp_min = clamp_min

        if not take_log and clamp_min is not None:
            raise ValueError("clamp_min is specified, but is not used.")

        self.criterion = nn.L1Loss(reduction=reduction)
        self.transform = aT.MelSpectrogram(
            sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            pad=pad,
            n_mels=n_mels,
            window_fn=window_fn,
            power=power,
            normalized=normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=onesided,
            norm=norm,
            mel_scale=mel_scale,
        )

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of MelSpectrogramL1Loss.

        Args:
            input (torch.Tensor): Waveform of shape (*, length).
            target (torch.Tensor): Waveform of shape (*, length).

        """
        clamp_min = self.clamp_min

        input = self.transform(input)
        target = self.transform(target)

        if self.take_log:
            if clamp_min is not None:
                input = torch.clamp(input, min=clamp_min)
                target = torch.clamp(target, min=clamp_min)

            input = torch.log(input)
            target = torch.log(target)

        loss = self.criterion(input, target)

        return loss
