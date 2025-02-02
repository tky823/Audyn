"""Criterion for Band-split RNN."""

from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "SpectrogramL1Loss",
    "WaveformL1Loss",
    "SpectrogramL1SNR",
    "WaveformL1SNR",
]


class SpectrogramL1Loss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()

        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduction = self.reduction

        loss_real = torch.abs(input.real - target.real)
        loss_imag = torch.abs(input.imag - target.imag)

        if reduction == "mean":
            loss = torch.mean(loss_real + loss_imag)
        elif reduction == "sum":
            loss = torch.sum(loss_real + loss_imag)
        elif reduction == "none":
            loss = loss_real + loss_imag
        else:
            raise ValueError(f"Invalid reduction {reduction} is found.")

        return loss


class WaveformL1Loss(nn.Module):

    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: Optional[bool] = False,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", window)
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.length = length
        self.return_complex = return_complex
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduction = self.reduction

        x_input = self._istft(input)
        x_target = self._istft(target)
        loss = torch.abs(x_input - x_target)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction {reduction} is found.")

        return loss

    def _istft(self, spectrogram: torch.Tensor) -> torch.Tensor:
        window = self.window

        *batch_shape, n_bins, n_frames = spectrogram.size()

        if isinstance(window, torch.Tensor):
            window = window.to(spectrogram.device)

        kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "length": self.length,
            "return_complex": self.return_complex,
        }

        x = spectrogram.contiguous()
        x = x.view(-1, n_bins, n_frames)
        x = torch.istft(x, **kwargs)
        waveform = x.view(*batch_shape, -1)

        return waveform


class SpectrogramL1SNR(nn.Module):

    def __init__(self, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()

        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduction = self.reduction
        eps = self.eps

        loss_real = torch.abs(input.real - target.real)
        loss_imag = torch.abs(input.imag - target.imag)
        loss_real = loss_real.sum(dim=(-2, -1))
        loss_imag = loss_imag.sum(dim=(-2, -1))
        target_real = torch.abs(target.real)
        target_imag = torch.abs(target.imag)
        target_real = target_real.sum(dim=(-2, -1))
        target_imag = target_imag.sum(dim=(-2, -1))

        loss_real = 10 * (torch.log10(loss_real + eps) - torch.log10(target_real + eps))
        loss_imag = 10 * (torch.log10(loss_imag + eps) - torch.log10(target_imag + eps))

        if reduction == "mean":
            loss = torch.mean(loss_real + loss_imag)
        elif reduction == "sum":
            loss = torch.sum(loss_real + loss_imag)
        elif reduction == "none":
            loss = loss_real + loss_imag
        else:
            raise ValueError(f"Invalid reduction {reduction} is found.")

        return loss


class WaveformL1SNR(nn.Module):

    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: Optional[bool] = None,
        length: Optional[int] = None,
        return_complex: Optional[bool] = False,
        reduction: str = "mean",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", window)
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        self.length = length
        self.return_complex = return_complex
        self.reduction = reduction
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        reduction = self.reduction
        eps = self.eps

        x_input = self._istft(input)
        x_target = self._istft(target)
        loss = torch.abs(x_input - x_target)
        loss = loss.sum(dim=-1)
        x_target = torch.abs(x_target)
        x_target = x_target.sum(dim=-1)

        loss = 10 * (torch.log10(loss + eps) - torch.log10(x_target + eps))

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            pass
        else:
            raise ValueError(f"Invalid reduction {reduction} is found.")

        return loss

    def _istft(self, spectrogram: torch.Tensor) -> torch.Tensor:
        window = self.window

        *batch_shape, n_bins, n_frames = spectrogram.size()

        if isinstance(window, torch.Tensor):
            window = window.to(spectrogram.device)

        kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": window,
            "center": self.center,
            "normalized": self.normalized,
            "onesided": self.onesided,
            "length": self.length,
            "return_complex": self.return_complex,
        }

        x = spectrogram.contiguous()
        x = x.view(-1, n_bins, n_frames)
        x = torch.istft(x, **kwargs)
        waveform = x.view(*batch_shape, -1)

        return waveform
