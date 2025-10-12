import warnings
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as aT

from ..functional.melspectrogram import melscale_fbanks

__all__ = [
    "LAIONCLAPAudioEncoder2023WaveformPad",
    "LAIONAudioEncoder2023WaveformPad",
    "LAIONCLAPAudioEncoder2023MelSpectrogram",
    "LAIONAudioEncoder2023MelSpectrogram",
    "LAIONCLAPAudioEncoder2023MelSpectrogramFusion",
    "LAIONAudioEncoder2023MelSpectrogramFusion",
    "MicrosoftCLAPAudioEncoder2023WaveformPad",
    "MicrosoftAudioEncoder2023WaveformPad",
    "MicrosoftCLAPAudioEncoder2023MelSpectrogram",
    "MicrosoftAudioEncoder2023MelSpectrogram",
]


class LAIONCLAPAudioEncoder2023WaveformPad(nn.Module):
    """Waveform padding for LAIONCLAPAudioEncoder2023.

    Args:
        pad_mode (str): Padding mode. ``"replicate+constant"`` and ``"replicate"``
            are supported.
        min_length (int, optional): Required length.

    .. note::

        Unlike ``MicrosoftCLAPAudioEncoder2023WaveformPad``, ``pad_mode`` defaults to
        ``"replicate+constant"``.

    """

    def __init__(
        self, pad_mode: str = "replicate+constant", min_length: Optional[int] = None
    ) -> None:
        super().__init__()

        self.pad_mode = pad_mode
        self.min_length = min_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pad_mode = self.pad_mode
        min_length = self.min_length

        *batch_shape, length = waveform.size()

        if min_length is not None and length < min_length:
            waveform = waveform.view(-1, 1, length)

            if pad_mode == "replicate+constant":
                repeats = min_length // length
            elif pad_mode == "replicate":
                repeats = (min_length - 1) // length + 1
            else:
                raise ValueError(f"Invalid {pad_mode} is found as pad_mode.")

            waveform = waveform.repeat(1, 1, repeats)

            # trim or pad
            waveform = F.pad(waveform, (0, min_length - waveform.size(-1)))

            waveform = waveform.view(*batch_shape, min_length)

        return waveform

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> "LAIONCLAPAudioEncoder2023WaveformPad":
        """Build predefined LAIONCLAPAudioEncoder2023WaveformPad.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model.

        Examples:

            >>> import torch
            >>> from audyn.transforms import LAIONCLAPAudioEncoder2023WaveformPad
            >>> torch.manual_seed(0)
            >>> waveform = torch.randn((200000,))
            >>> padding = LAIONCLAPAudioEncoder2023WaveformPad.build_from_pretrained("laion-clap-htsat-fused")
            >>> waveform = padding(waveform)
            >>> print(waveform.size())
            torch.Size([480000])

        .. note::

            Supported pretrained model names are
                - laion-clap-htsat-fused

        """  # noqa: E501
        if pretrained_model_name_or_path == "laion-clap-htsat-fused":
            transform = cls(
                pad_mode="replicate+constant",
                min_length=10 * 48000,
            )
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not supported as "
                "pretrained_model_name_or_path."
            )

        return transform


class LAIONCLAPAudioEncoder2023MelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for LAIONCLAPAudioEncoder2023.

    For the details of arguments, see ``torchaudio.transforms.MelSpectrogram``.

    .. note::

        Value of ``mel_scale`` depends on ``truncation`` in original implementation.
        See https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/clap/feature_extraction_clap.py#L153-L163.

    Examples:

        >>> import torch
        >>> from audyn.transforms import LAIONCLAPAudioEncoder2023MelSpectrogram
        >>> torch.manual_seed(0)
        >>> waveform = torch.randn((48000,))
        >>> # truncation is "fusion"
        >>> melspectrogram_transform = LAIONCLAPAudioEncoder2023MelSpectrogram(norm=None, mel_scale="htk")
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> melspectrogram.size()
        torch.Size([64, 101])
        >>> # truncation is NOT "fusion" (default by transformers)
        >>> # https://github.com/huggingface/transformers/blob/6bc0fbcfa7acb6ac4937e7456a76c2f7975fefec/src/transformers/models/clap/feature_extraction_clap.py#L96
        >>> melspectrogram_transform = LAIONCLAPAudioEncoder2023MelSpectrogram(norm="slaney", mel_scale="slaney")
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> melspectrogram.size()
        torch.Size([64, 101])

    """  # noqa: E501

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 480,
        f_min: float = 50,
        f_max: float = 14000,
        pad: int = 0,
        n_mels: int = 64,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        fb_dtype: Optional[torch.dtype] = None,
    ) -> None:
        is_float64_fb = fb_dtype is torch.float64

        if is_float64_fb:
            if wkwargs is None:
                wkwargs = {}

            wkwargs["dtype"] = torch.float64

        super().__init__(
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

        self.amplitude_to_db = aT.AmplitudeToDB()
        self.fb_dtype = fb_dtype

        if is_float64_fb:
            fb = melscale_fbanks(
                n_fft // 2 + 1,
                f_min,
                f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm=norm,
                mel_scale=mel_scale,
                dtype=torch.float64,
            )

            self.mel_scale.fb = fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform_dtype = waveform.dtype
        waveform = waveform.to(self.mel_scale.fb.dtype)

        spectrogram = super().forward(waveform)
        output = self.amplitude_to_db(spectrogram)
        output = output.to(waveform_dtype)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> "LAIONCLAPAudioEncoder2023MelSpectrogram":
        """Build predefined LAIONCLAPAudioEncoder2023MelSpectrogram.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model.

        Examples:

            >>> import torch
            >>> from audyn.transforms import LAIONCLAPAudioEncoder2023MelSpectrogram
            >>> torch.manual_seed(0)
            >>> waveform = torch.randn((48000,))
            >>> melspectrogram_transform = LAIONCLAPAudioEncoder2023MelSpectrogram.build_from_pretrained("laion-clap-htsat-fused")
            >>> melspectrogram = melspectrogram_transform(waveform)
            >>> melspectrogram.size()
            torch.Size([64, 101])

        .. note::

            Supported pretrained model names are
                - laion-clap-htsat-fused

        """  # noqa: E501
        if pretrained_model_name_or_path == "laion-clap-htsat-fused":
            transform = cls(
                sample_rate=48000,
                n_fft=1024,
                win_length=1024,
                hop_length=480,
                f_min=50,
                f_max=14000,
                pad=0,
                n_mels=64,
                window_fn=torch.hann_window,
                power=2.0,
                normalized=False,
                wkwargs=None,
                center=True,
                pad_mode="reflect",
                onesided=None,
                norm=None,
                mel_scale="htk",
                fb_dtype=torch.float64,
            )
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not supported as "
                "pretrained_model_name_or_path."
            )

        return transform


class LAIONCLAPAudioEncoder2023MelSpectrogramFusion(nn.Module):
    """Mel-spectrogram fusion for LAIONCLAPAudioEncoder2023.

    Args:
        chunk_size (int): Chunk size. Default: ``1001``.
        num_chunks (int): Number of chunks. Default: ``3``.
        dim (int): Dimension to stack. Default: ``-3``.
        prepend_resampled_chunk (bool): If ``True``, resampled chunk of Mel-spectrogram
            is prepended.
        sample_wise (bool): If ``True``, masking is applied per sample.

    """

    def __init__(
        self,
        chunk_size: int = 1001,
        num_chunks: int = 3,
        dim: int = -3,
        prepend_resampled_chunk: bool = True,
        pad_mode: str = "replicate+constant",
        sample_wise: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()

        assert pad_mode in ["replicate", "replicate+constant"]

        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.dim = dim
        self.prepend_resampled_chunk = prepend_resampled_chunk
        self.pad_mode = pad_mode
        self.sample_wise = sample_wise
        self.generator = torch.Generator()

        self.generator.manual_seed(seed)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks

        assert spectrogram.dim() >= 2

        *batch_shape, n_bins, n_frames = spectrogram.size()
        spectrogram = spectrogram.view(-1, n_bins, n_frames)

        if n_frames >= chunk_size:
            if self.sample_wise:
                stacked_chunks = [[] for _ in range(num_chunks)]

                for _spectrogram in spectrogram:
                    _spectrogram = self._sample_chunks(_spectrogram)

                    for chunk_idx in range(num_chunks):
                        stacked_chunks[chunk_idx].append(_spectrogram[chunk_idx])

                for chunk_idx in range(num_chunks):
                    stacked_chunks[chunk_idx] = torch.stack(stacked_chunks[chunk_idx], dim=0)
                    stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                        *batch_shape, n_bins, chunk_size
                    )
            else:
                stacked_chunks = self._sample_chunks(spectrogram)

                for chunk_idx in range(num_chunks):
                    stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                        *batch_shape, n_bins, chunk_size
                    )
        else:
            warnings.warn(
                f"Number of frames {n_frames} is shorter than required chunk_size {chunk_size}.",
                UserWarning,
                stacklevel=2,
            )

            stacked_chunks = self._pad_chunks(spectrogram)

            for chunk_idx in range(num_chunks):
                stacked_chunks[chunk_idx] = stacked_chunks[chunk_idx].view(
                    *batch_shape, n_bins, chunk_size
                )

        output = torch.stack(stacked_chunks, dim=self.dim)

        if self.prepend_resampled_chunk:
            resampled_chunk = F.interpolate(
                spectrogram.view(-1, 1, n_bins, n_frames),
                size=(n_bins, chunk_size),
                mode="bilinear",
                align_corners=False,
            )
            resampled_chunk = resampled_chunk.view(*batch_shape, n_bins, chunk_size)
            resampled_chunk = resampled_chunk.unsqueeze(dim=self.dim)
            output = torch.cat([resampled_chunk, output], dim=self.dim)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> "LAIONCLAPAudioEncoder2023MelSpectrogramFusion":
        """Build predefined LAIONCLAPAudioEncoder2023MelSpectrogramFusion.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model.

        Examples:

            >>> import torch
            >>> from audyn.transforms import LAIONCLAPAudioEncoder2023MelSpectrogramFusion
            >>> torch.manual_seed(0)
            >>> fusion_transform = LAIONCLAPAudioEncoder2023MelSpectrogramFusion.build_from_pretrained("laion-clap-htsat-fused")
            >>> # unbatched
            >>> melspectrogram = torch.randn((64, 2001))
            >>> fused_melspectrogram = fusion_transform(melspectrogram)
            >>> fused_melspectrogram.size()
            torch.Size([4, 64, 001])
            >>> # batched
            >>> melspectrogram = torch.randn((2, 64, 2001))
            >>> fused_melspectrogram = fusion_transform(melspectrogram)
            >>> fused_melspectrogram.size()
            torch.Size([2, 4, 64, 1001])

        .. note::

            Supported pretrained model names are
                - laion-clap-htsat-fused

        """  # noqa: E501
        if pretrained_model_name_or_path == "laion-clap-htsat-fused":
            transform = cls(
                chunk_size=1001,
                num_chunks=3,
                dim=-3,
                prepend_resampled_chunk=True,
                pad_mode="replicate+constant",
                sample_wise=True,
            )
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not supported as "
                "pretrained_model_name_or_path."
            )

        return transform

    def _sample_chunks(self, spectrogram: torch.Tensor) -> List[torch.Tensor]:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks
        n_frames = spectrogram.size(-1)
        valid_max_idx = n_frames - chunk_size

        if valid_max_idx < num_chunks:
            # trim trailing
            spectrogram = F.pad(spectrogram, (0, chunk_size - n_frames))
            spectrograms = spectrogram.expand(num_chunks, *spectrogram.size())
            spectrograms = torch.unbind(spectrograms, dim=0)
            spectrograms = list(spectrograms)
        else:
            min_idx = 0
            spectrograms = []

            for chunk_idx in range(num_chunks):
                max_idx = int(valid_max_idx * (chunk_idx + 1) / num_chunks)

                if self.training:
                    start_idx = torch.randint(min_idx, max_idx, (), generator=self.generator)
                    start_idx = start_idx.item()
                else:
                    start_idx = (min_idx + max_idx) // 2

                end_idx = start_idx + chunk_size
                _, _spectrogram, _ = torch.split(
                    spectrogram,
                    [start_idx, chunk_size, n_frames - end_idx],
                    dim=-1,
                )
                spectrograms.append(_spectrogram)

                min_idx = max_idx

        return spectrograms

    def _pad_chunks(self, spectrogram: torch.Tensor) -> List[torch.Tensor]:
        chunk_size = self.chunk_size
        num_chunks = self.num_chunks
        pad_mode = self.pad_mode
        *batch_shape, n_bins, n_frames = spectrogram.size()

        spectrogram = spectrogram.view(-1, n_bins, n_frames)

        if pad_mode == "replicate+constant":
            repeats = chunk_size // n_frames
        elif pad_mode == "replicate":
            repeats = (chunk_size - 1) // n_frames + 1
        else:
            raise ValueError(f"Invalid {pad_mode} is found as pad_mode.")

        spectrogram = spectrogram.repeat(1, 1, repeats)

        # trim or pad
        spectrogram = F.pad(spectrogram, (0, chunk_size - spectrogram.size(-1)))
        spectrogram = spectrogram.view(*batch_shape, n_bins, -1)
        spectrograms = spectrogram.expand(num_chunks, *spectrogram.size())
        spectrograms = torch.unbind(spectrograms, dim=0)
        spectrograms = list(spectrograms)

        return spectrograms


class MicrosoftCLAPAudioEncoder2023WaveformPad(LAIONCLAPAudioEncoder2023WaveformPad):
    """Waveform padding for MicrosoftCLAPAudioEncoder2023.

    Args:
        pad_mode (str): Padding mode. ``"replicate+constant"`` and ``"replicate"``
            are supported. Default: ``"replicate"``.
        min_length (int, optional): Required length.

    .. note::

        Unlike ``LAIONCLAPAudioEncoder2023WaveformPad``, ``pad_mode`` defaults to
        ``"replicate"``.

    """

    def __init__(self, pad_mode: str = "replicate", min_length: Optional[int] = None) -> None:
        super().__init__()

        self.pad_mode = pad_mode
        self.min_length = min_length

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pad_mode = self.pad_mode
        min_length = self.min_length

        *batch_shape, length = waveform.size()

        if min_length is not None and length < min_length:
            waveform = waveform.view(-1, 1, length)

            if pad_mode == "replicate+constant":
                repeats = min_length // length
            elif pad_mode == "replicate":
                repeats = (min_length - 1) // length + 1
            else:
                raise ValueError(f"Invalid {pad_mode} is found as pad_mode.")

            waveform = waveform.repeat(1, 1, repeats)

            # trim or pad
            waveform = F.pad(waveform, (0, min_length - waveform.size(-1)))

            waveform = waveform.view(*batch_shape, min_length)

        return waveform

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sample_rate: int = 44100,
    ) -> "MicrosoftCLAPAudioEncoder2023WaveformPad":
        """Build predefined MicrosoftCLAPAudioEncoder2023WaveformPad.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model.
            sample_rate (int): Sampling rate. Default: ``44100``.

        Examples:

            >>> import torch
            >>> from audyn.transforms import MicrosoftCLAPAudioEncoder2023WaveformPad
            >>> torch.manual_seed(0)
            >>> waveform = torch.randn((100000,))
            >>> padding = MicrosoftCLAPAudioEncoder2023WaveformPad.build_from_pretrained("microsoft-clap-2023")
            >>> waveform = padding(waveform)
            >>> print(waveform.size())
            torch.Size([224000])

        .. note::

            Supported pretrained model names are
                - microsoft-clap-2023

        """  # noqa: E501
        if pretrained_model_name_or_path == "microsoft-clap-2023":
            transform = cls(
                pad_mode="replicate",
                min_length=7 * sample_rate,
            )
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not supported as "
                "pretrained_model_name_or_path."
            )

        return transform


class MicrosoftCLAPAudioEncoder2023MelSpectrogram(aT.MelSpectrogram):
    """Mel-spectrogram transform for MicrosoftCLAPAudioEncoder2023.

    For the details of arguments, see ``torchaudio.transforms.MelSpectrogram``.
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 320,
        f_min: float = 50,
        f_max: float = 14000,
        pad: int = 0,
        n_mels: int = 64,
        window_fn: Callable[[int], torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[Dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
        onesided: Optional[bool] = None,
        norm: Optional[str] = None,
        mel_scale: str = "htk",
        max_length: Optional[int] = None,
        max_duration: Optional[float] = None,
        fb_dtype: Optional[torch.dtype] = None,
        seed: int = 0,
    ) -> None:
        if max_length is not None:
            if max_duration is not None:
                assert max_length == int(sample_rate * max_duration), (
                    "length and duration is not consistent."
                )
        elif max_duration is not None:
            max_length = int(sample_rate * max_duration)

        is_float64_fb = fb_dtype is torch.float64

        if is_float64_fb:
            if wkwargs is None:
                wkwargs = {}

            wkwargs["dtype"] = torch.float64

        super().__init__(
            sample_rate=sample_rate,
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

        self.amplitude_to_db = aT.AmplitudeToDB()
        self.max_length = max_length
        self.fb_dtype = fb_dtype
        self.generator = torch.Generator()

        self.generator.manual_seed(seed)

        if is_float64_fb:
            fb = melscale_fbanks(
                n_fft // 2 + 1,
                f_min,
                f_max,
                n_mels=n_mels,
                sample_rate=sample_rate,
                norm=norm,
                mel_scale=mel_scale,
                dtype=torch.float64,
            )

            self.mel_scale.fb = fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        max_length = self.max_length

        waveform_dtype = waveform.dtype

        if max_length is not None and waveform.size(-1) > max_length:
            if self.training:
                start_idx = torch.randint(
                    0, waveform.size(-1) - max_length, (), generator=self.generator
                )
                start_idx = start_idx.item()
            else:
                start_idx = (waveform.size(-1) - max_length) // 2

            end_idx = start_idx + max_length
            _, waveform, _ = torch.split(
                waveform, [start_idx, max_length, waveform.size(-1) - end_idx], dim=-1
            )

        waveform = waveform.to(self.mel_scale.fb.dtype)
        spectrogram = super().forward(waveform)
        output = self.amplitude_to_db(spectrogram)
        output = output.to(waveform_dtype)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        sample_rate: int = 32000,
        f_min: float = 50,
        f_max: float = 14000,
        max_length: int = 308700,
    ) -> "MicrosoftCLAPAudioEncoder2023MelSpectrogram":
        """Build predefined MicrosoftCLAPAudioEncoder2023MelSpectrogram.

        Args:
            pretrained_model_name_or_path (str): Name of pretrained model.
            sample_rate (int): Sampling rate. Default: ``32000``.
            f_min (float): Minimum frequency of Mel-spectrogram. Default: ``50``.
            f_max (float): Maximum frequency of Mel-spectrogram. Default: ``14000``.
            max_length (int): Maximum length of waveform. Default: ``308700``.

        Examples:

            >>> import torch
            >>> from audyn.transforms import MicrosoftCLAPAudioEncoder2023MelSpectrogram
            >>> torch.manual_seed(0)
            >>> waveform = torch.randn((224000,))
            >>> melspectrogram_transform = MicrosoftCLAPAudioEncoder2023MelSpectrogram.build_from_pretrained("microsoft-clap-2023")
            >>> melspectrogram = melspectrogram_transform(waveform)
            >>> melspectrogram.size()
            torch.Size([64, 701])

        .. note::

            Supported pretrained model names are
                - microsoft-clap-2023

        .. note::

            According to official paper, sampling rate and max frequency are set to 44100Hz and 8000Hz.
            However, Mel-spectrograms seem to be computed by 32000Hz and max frequency to be 14000Hz
            in official implementation. See also https://github.com/microsoft/CLAP/issues/38.

        """  # noqa: E501
        if pretrained_model_name_or_path == "microsoft-clap-2023":
            if sample_rate != 32000:
                raise ValueError(
                    f"sample_rate should be 32000 for {pretrained_model_name_or_path}."
                )

            if f_min != 50:
                raise ValueError(f"f_min should be 50 for {pretrained_model_name_or_path}.")

            if f_max != 14000:
                raise ValueError(f"f_max should be 14000 for {pretrained_model_name_or_path}.")

            if max_length != 308700:
                raise ValueError(
                    f"max_length should be 308700 for {pretrained_model_name_or_path}."
                )

            transform = cls(
                sample_rate=sample_rate,
                n_fft=1024,
                win_length=1024,
                hop_length=320,
                f_min=f_min,
                f_max=f_max,
                pad=0,
                n_mels=64,
                window_fn=torch.hann_window,
                power=2.0,
                normalized=False,
                wkwargs=None,
                center=True,
                pad_mode="reflect",
                onesided=None,
                norm="slaney",
                mel_scale="slaney",
                max_length=max_length,
                max_duration=None,
                fb_dtype=torch.float64,
            )
        else:
            raise ValueError(
                f"{pretrained_model_name_or_path} is not supported as "
                "pretrained_model_name_or_path."
            )

        return transform


class LAIONAudioEncoder2023WaveformPad(LAIONCLAPAudioEncoder2023WaveformPad):
    """Alias of LAIONCLAPAudioEncoder2023WaveformPad."""


class LAIONAudioEncoder2023MelSpectrogram(LAIONCLAPAudioEncoder2023MelSpectrogram):
    """Alias of LAIONCLAPAudioEncoder2023MelSpectrogram."""


class LAIONAudioEncoder2023MelSpectrogramFusion(LAIONCLAPAudioEncoder2023MelSpectrogramFusion):
    """Alias of LAIONCLAPAudioEncoder2023MelSpectrogramFusion."""


class MicrosoftAudioEncoder2023WaveformPad(MicrosoftCLAPAudioEncoder2023WaveformPad):
    """Alias of MicrosoftCLAPAudioEncoder2023WaveformPad."""


class MicrosoftAudioEncoder2023MelSpectrogram(MicrosoftCLAPAudioEncoder2023MelSpectrogram):
    """Alias of MicrosoftCLAPAudioEncoder2023MelSpectrogram."""
