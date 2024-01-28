import functools
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torchaudio.compliance.kaldi as aCK
from packaging import version

__all__ = ["KaldiMelSpectrogram"]

IS_TORCH_LT_2_0_0 = version.parse(torch.__version__) < version.parse("2.0.0")


class KaldiMelSpectrogram(nn.Module):
    """Wrapper class of Mel-spectrogram transform using torchaudio.compliance.kaldi.fbank.

    Args:
        sample_rate (int): Sampling rate called as sample_frequency in
            torchaudio.compliance.kaldi.fbank.
        win_length (int): Window length. ``win_length / sample_rate`` should be equal to
            ``frame_length`` in torchaudio.compliance.kaldi.fbank.
        hop_length (int): Hop length. ``hop_length / sample_rate`` should be equal to
            ``frame_shift`` in torchaudio.compliance.kaldi.fbank.
        f_min (float): Minimum frequency called as low_freq in torchaudio.compliance.kaldi.fbank.
        f_max (float): Maximum frequency called as high_freq in torchaudio.compliance.kaldi.fbank.
        n_mels (int): Number of mel filterbanks called as num_mel_bins in
            torchaudio.compliance.kaldi.fbank.
        power (float, optional): Exponent for the magnitude spectrogram. Only 1 and 2 are
            supported.
        fbank_kwargs (dict, optional): Keyword arguments given to
            torchaudio.compliance.kaldi.fbank. Some values should be compatible with arguments
            defined above.

    """

    def __init__(
        self,
        sample_rate: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = None,
        f_max: float = None,
        n_mels: Optional[int] = None,
        power: Optional[float] = None,
        fbank_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if fbank_kwargs is None:
            fbank_kwargs = {}

        if "sample_frequency" in fbank_kwargs:
            sample_frequency = fbank_kwargs["sample_frequency"]

            assert sample_rate == sample_frequency, (
                f"sample_rate ({sample_rate}) should be equal to "
                f"sample_frequency ({sample_frequency}) in fbank_kwargs."
            )
        else:
            fbank_kwargs["sample_frequency"] = sample_rate

        if win_length is not None:
            if "frame_length" in fbank_kwargs:
                frame_length = fbank_kwargs["frame_length"]

                assert win_length / sample_rate == frame_length, (
                    f"win_length ({win_length}) should be compatible with "
                    f"frame_length ({frame_length}) in fbank_kwargs."
                )
            else:
                fbank_kwargs["frame_length"] = win_length / sample_rate

        if hop_length is not None:
            if "frame_shift" in fbank_kwargs:
                frame_shift = fbank_kwargs["frame_shift"]

                assert hop_length / sample_rate == frame_shift, (
                    f"hop_length ({hop_length}) should be compatible with "
                    f"frame_shift ({frame_shift}) in fbank_kwargs."
                )
            else:
                fbank_kwargs["frame_shift"] = hop_length / sample_rate

        if f_min is not None:
            if "low_freq" in fbank_kwargs:
                low_freq = fbank_kwargs["low_freq"]

                assert f_min == low_freq, (
                    f"f_min ({f_min}) should be compatible with "
                    f"low_freq ({low_freq}) in fbank_kwargs."
                )
            else:
                fbank_kwargs["low_freq"] = f_min

        if f_max is not None:
            if "high_freq" in fbank_kwargs:
                high_freq = fbank_kwargs["high_freq"]

                assert f_max == high_freq, (
                    f"f_max ({f_max}) should be compatible with "
                    f"high_freq ({high_freq}) in fbank_kwargs."
                )
            else:
                fbank_kwargs["high_freq"] = f_max

        if n_mels is not None:
            if "num_mel_bins" in fbank_kwargs:
                num_mel_bins = fbank_kwargs["num_mel_bins"]

                assert n_mels == num_mel_bins, (
                    f"n_mels ({n_mels}) should be equal to "
                    f"num_mel_bins ({num_mel_bins}) in fbank_kwargs."
                )
            else:
                fbank_kwargs["num_mel_bins"] = n_mels

        if power is not None:
            if "use_power" in fbank_kwargs:
                use_power = fbank_kwargs["use_power"]

                assert type(use_power) is bool

                if use_power:
                    assert power == 2, "`power` should be 2 when use_power=True."
                else:
                    assert power == 1, "`power` should be 1 when use_power=False."
            else:
                if power == 1:
                    use_power = False
                elif power == 2:
                    use_power = True
                else:
                    raise ValueError(f"power={power} is not supported. Use 1.0 or 2.0.")

                fbank_kwargs["use_power"] = use_power

        self.fbank_kwargs = fbank_kwargs

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram transform.

        Args:
            waveform (torch.Tensor): Waveform of shape (batch_size, timesteps)
                or (batch_size, 1, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (batch_size, n_mels, n_frames)
                or (batch_size, 1, n_mels, n_frames).

        """
        fbank_kwargs = self.fbank_kwargs

        if IS_TORCH_LT_2_0_0:
            spectrogram = self._sequential_fbank(waveform, **fbank_kwargs)
        else:
            spectrogram = self._parallel_fbank(waveform, **fbank_kwargs)

        return spectrogram

    @staticmethod
    def _sequential_fbank(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply _fbank_fn sequentially.

        Args:
            waveform (torch.Tensor): Waveform of shape (batch_size, timesteps)
                or (batch_size, 1, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (batch_size, n_mels, n_frames)
                or (batch_size, 1, n_mels, n_frames).

        """
        spectrogram = []

        for _waveform in waveform:
            _spectrogram = _fbank_fn(_waveform, **kwargs)
            spectrogram.append(_spectrogram)

        spectrogram = torch.stack(spectrogram, dim=0)

        return spectrogram

    @staticmethod
    def _parallel_fbank(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply _fbank_fn via torch.vmap.

        Args:
            waveform (torch.Tensor): Waveform of shape (batch_size, timesteps)
                or (batch_size, 1, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (batch_size, n_mels, n_frames)
                or (batch_size, 1, n_mels, n_frames).

        """
        vfbank_fn = torch.vmap(functools.partial(_fbank_fn, **kwargs))
        spectrogram = vfbank_fn(waveform)

        return spectrogram


def _fbank_fn(waveform: torch.Tensor, **kwargs) -> torch.Tensor:
    # torchaudio.compliance.kaldi.fbank expects tensor
    # of shape (n_channels, time).
    n_dims = waveform.dim()

    if n_dims == 1:
        waveform = waveform.unsqueeze(dim=0)
    elif n_dims == 2:
        if waveform.size(0) != 1:
            raise ValueError("If 2D waveform is given, number of channels at dim 0 should be 1.")
    else:
        raise ValueError(f"Waveform should be 1D or 2D tensor, but {n_dims}D tensor is used.")

    spectrogram = aCK.fbank(waveform, **kwargs)
    spectrogram = spectrogram.transpose(1, 0)

    if n_dims == 2:
        spectrogram = spectrogram.unsqueeze(dim=-3)

    return spectrogram
