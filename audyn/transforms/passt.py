from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.compliance.kaldi as aCK
import torchaudio.transforms as aT

from ..amp import autocast, get_autocast_device_type
from ..utils.data.audioset.passt import mean as _audioset_mean
from ..utils.data.audioset.passt import std as _audioset_std


class PaSSTMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_fft: int = 1024,
        win_length: Optional[int] = None,
        hop_length: int = 320,
        f_min: float = 0,
        f_max: Optional[float] = None,
        n_mels: Optional[int] = None,
        freq_aug_param: Optional[Tuple[int, int]] = None,
        freq_mask_param: Optional[int] = None,
        time_mask_param: Optional[int] = None,
        mean: float = 0,
        std: float = 1,
        take_log: bool = True,
        eps: float = 0.00001,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if win_length is None:
            win_length = 800

        if freq_aug_param is None:
            freq_aug_param = (1, 1000)
        else:
            assert len(freq_aug_param) == 2

        if f_max is None:
            f_max = sample_rate // 2 - freq_aug_param[1] // 2

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.f_min = f_min
        self.f_max = f_max
        self.freq_aug_param = freq_aug_param
        self.n_mels = n_mels
        self.mean = mean
        self.std = std
        self.take_log = take_log
        self.eps = eps

        self.fbank_kwargs = {
            "vtln_low": 100.0,
            "vtln_high": -500.0,
            "vtln_warp_factor": 1.0,
        }

        if freq_mask_param is None:
            self.frequency_masking = None
        else:
            self.frequency_masking = aT.FrequencyMasking(freq_mask_param)

        if time_mask_param is None:
            self.time_masking = None
        else:
            self.time_masking = aT.TimeMasking(time_mask_param)

        self.register_buffer("preemphasis_coefficient", torch.tensor([-0.97, 1]), persistent=False)
        self.register_buffer(
            "window", torch.hann_window(win_length, periodic=False), persistent=False
        )

        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Mel-spectrogram transform.

        Args:
            waveform (torch.Tensor): Waveform of shape (batch_size, timesteps)
                or (batch_size, 1, timesteps).

        Returns:
            torch.Tensor: Mel-spectrogram of shape (batch_size, n_mels, n_frames)
                or (batch_size, 1, n_mels, n_frames).

        """
        mean = self.mean
        std = self.std
        take_log = self.take_log

        preemphasis_coefficient = self.preemphasis_coefficient.view(1, 1, -1)
        window = self.window.to(waveform.device)

        *batch_shape, timesteps = waveform.size()

        waveform = waveform.view(-1, 1, timesteps)
        waveform = F.conv1d(waveform, preemphasis_coefficient)
        waveform = waveform.view(-1, timesteps - 1)

        spectrogram = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=True,
            normalized=False,
            window=window,
            return_complex=True,
        )
        spectrogram = torch.abs(spectrogram) ** 2

        if self.training:
            f_min_offset, f_max_offset = self.freq_aug_param
            # f_min
            offset = torch.randint(0, f_min_offset, (), generator=self.generator)
            f_min = self.f_min + offset.item()
            # f_max
            offset = f_max_offset // 2 - torch.randint(
                0, f_max_offset, (), generator=self.generator
            )
            f_max = self.f_max + offset.item()
        else:
            f_min = self.f_min
            f_max = self.f_max

        mel_basis, _ = aCK.get_mel_banks(
            self.n_mels,
            self.n_fft,
            self.sample_rate,
            f_min,
            f_max,
            **self.fbank_kwargs,
        )
        mel_basis = F.pad(mel_basis, (0, 1), mode="constant", value=0)
        mel_basis = mel_basis.to(spectrogram.device)
        device_type = get_autocast_device_type(mel_basis)

        with autocast(device_type, enabled=False):
            spectrogram = torch.matmul(mel_basis, spectrogram)

        spectrogram = torch.log(spectrogram + self.eps)

        if self.training:
            if self.frequency_masking is not None:
                spectrogram = self.frequency_masking(spectrogram)

            if self.time_masking is not None:
                spectrogram = self.time_masking(spectrogram)

        spectrogram = (spectrogram - mean) / std

        if not take_log:
            spectrogram = torch.exp(spectrogram)

        spectrogram = spectrogram.view(*batch_shape, self.n_mels, -1)

        return spectrogram

    @classmethod
    def build_from_default_config(
        cls,
        dataset: str,
        sample_rate: Optional[int] = None,
        n_mels: Optional[int] = None,
        freq_mask_param: Optional[int] = None,
        time_mask_param: Optional[int] = None,
        mean: Optional[float] = None,
        std: Optional[float] = None,
        take_log: bool = True,
    ) -> "PaSSTMelSpectrogram":
        """Dataset-aware factory method for PaSSTMelSpectrogram.

        Args:
            dataset (str): Dataset name. Now, ``audioset`` is available.

        Returns:
            PaSSTMelSpectrogram: Mel-spectrogram transform.

        """
        if dataset.lower() == "audioset":
            if sample_rate is None:
                sample_rate = 32000

            if n_mels is None:
                n_mels = 128

            if freq_mask_param is None:
                freq_mask_param = 48

            if time_mask_param is None:
                time_mask_param = 192

            if mean is None:
                mean = _audioset_mean

            if std is None:
                std = _audioset_std

            transform = cls(
                sample_rate,
                n_mels=n_mels,
                freq_mask_param=freq_mask_param,
                time_mask_param=time_mask_param,
                mean=mean,
                std=std,
                take_log=take_log,
            )
        else:
            raise ValueError(f"{dataset} is not supported as dataset.")

        return transform
