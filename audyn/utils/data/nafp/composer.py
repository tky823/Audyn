from typing import Any, Dict, Optional, Tuple

import torch
import torchaudio
from packaging import version

from ..composer import Composer

__all__ = [
    "NeuralAudioFingerprintingWaveformSliceComposer",
    "NeuralAudioFingerprintingSpectrogramAugmentationComposer",
    "NAFPWaveformSliceComposer",
    "NAFPSpectrogramAugmentationComposer",
]

IS_TORCHAUDIO_LT_2_1 = version.parse(torchaudio.__version__) < version.parse("2.1")


class NeuralAudioFingerprintingWaveformSliceComposer(Composer):
    """Composer to slice waveform and augmented one by fixed length or duration.

    Args:
        input_key (str): Key of tensor to slice waveforms.
        output_key (str): Key of tensor to store sliced waveforms.
        shifted_key (str, optional): Key of shifted waveforms.
        length (int, optional): Length of waveform slice.
        duration (float, optional): Duration of waveform slice.
        offset_length (int, optional): Max offset in length
        offset_duration (float, optional): Max offset in duration.
        sample_rate (int, optional): Sampling rate of waveform.
        training (bool): If ``True``, slice waveforms randomly.
        seed (int): Random seed. Default: ``0``.

    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        shifted_key: str,
        length: Optional[int] = None,
        duration: Optional[float] = None,
        offset_length: Optional[int] = None,
        offset_duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        training: bool = False,
        seed: int = 0,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        if length is None:
            assert duration is not None and sample_rate is not None

            length = int(sample_rate * duration)
        else:
            assert duration is None and sample_rate is None

        if offset_length is None:
            assert offset_duration is not None and sample_rate is not None

            offset_length = int(sample_rate * offset_duration)
        else:
            assert offset_duration is None and sample_rate is None

        self.input_key = input_key
        self.output_key = output_key
        self.shifted_key = shifted_key

        self.length = length
        self.duration = duration
        self.offset_length = offset_length
        self.offset_duration = offset_duration
        self.sample_rate = sample_rate
        self.training = training

        self.generator = None
        self.seed = seed

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key
        shifted_key = self.shifted_key
        length = self.length
        offset_length = self.offset_length
        training = self.training
        g = self.generator

        if g is None:
            seed = self.seed
            g = torch.Generator()
            g.manual_seed(seed)

        waveform = sample[input_key]

        *batch_shape, orig_length = waveform.size()
        waveform = waveform.view(-1, orig_length)

        if training:
            start_idx = torch.randint(0, orig_length - length - offset_length, (), generator=g)
            start_idx = start_idx.item()
        else:
            start_idx = (orig_length - length - offset_length) // 2

        end_idx = start_idx + length + offset_length

        _, sliced_waveform, _ = torch.split(
            waveform, [start_idx, length + offset_length, orig_length - end_idx], dim=-1
        )

        if training:
            start_idx = torch.randint(0, offset_length, (), generator=g)
            start_idx = start_idx.item()
        else:
            start_idx = offset_length

        end_idx = start_idx + length

        _, shifted_waveform, _ = torch.split(
            sliced_waveform, [start_idx, length, length + offset_length - end_idx], dim=-1
        )

        if training:
            start_idx = torch.randint(0, offset_length, (), generator=g)
        else:
            start_idx = 0

        end_idx = start_idx + length

        _, sliced_waveform, _ = torch.split(
            sliced_waveform, [start_idx, length, length + offset_length - end_idx], dim=-1
        )

        sample[output_key] = sliced_waveform.view(*batch_shape, length)
        sample[shifted_key] = shifted_waveform.view(*batch_shape, length)

        return sample


class NAFPWaveformSliceComposer(NeuralAudioFingerprintingWaveformSliceComposer):
    """Alias of NeuralAudioFingerprintingWaveformSliceComposer."""


class NeuralAudioFingerprintingSpectrogramAugmentationComposer(Composer):
    """Composer to augment spectrogram.

    Args:
        input_key (str): Key of spectrogram in given sample.
        output_key (str): Key of tensor to store augmented spectrogram.
        freq_mask_param (tuple): Parameter of frequency masking. Default: ``(0.01, 0.5)``.
        time_mask_param (tuple): Parameter of time masking. Default: ``(0.01, 0.5)``.
        patch_mask_param (tuple): Parameter of pacth masking (i.e., cut-out).
            Default: ``((0.01, 0.5), (0.01, 0.05))``.

    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        freq_mask_param: Tuple[float, float] = (0.01, 0.5),
        time_mask_param: Tuple[float, float] = (0.01, 0.5),
        patch_mask_param: Tuple[Tuple[float, float], Tuple[float, float]] = (
            (0.01, 0.5),
            (0.01, 0.5),
        ),
        training: bool = False,
        seed: int = 0,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.input_key = input_key
        self.output_key = output_key
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.patch_mask_param = patch_mask_param
        self.training = training

        self.generator = None
        self.seed = seed

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key
        freq_mask_param = self.freq_mask_param
        time_mask_param = self.time_mask_param
        patch_mask_param = self.patch_mask_param
        training = self.training

        specgram = sample[input_key]

        if self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        if training:
            *batch_shape, n_bins, n_frames = specgram.size()
            specgram = specgram.view(-1, n_bins, n_frames)

            # frequency mask
            start_bin, end_bin = _randrange(
                n_bins, param=freq_mask_param, generator=self.generator
            )
            bin_indices = torch.arange(n_bins)
            masking_mask = (start_bin <= bin_indices) & (bin_indices < end_bin)
            masking_mask = masking_mask.unsqueeze(dim=-1)
            specgram = specgram.masked_fill(masking_mask, 0)

            # time mask
            start_frame, end_frame = _randrange(
                n_frames, param=time_mask_param, generator=self.generator
            )
            frame_indices = torch.arange(n_frames)
            masking_mask = (start_frame <= frame_indices) & (frame_indices < end_frame)
            specgram = specgram.masked_fill(masking_mask, 0)

            # patch mask
            start_bin, end_bin = _randrange(
                n_bins, param=patch_mask_param[0], generator=self.generator
            )
            start_frame, end_frame = _randrange(
                n_frames, param=patch_mask_param[1], generator=self.generator
            )
            bin_indices = torch.arange(n_bins)
            frame_indices = torch.arange(n_frames)
            freq_masking_mask = (start_bin <= bin_indices) & (bin_indices < end_bin)
            time_masking_mask = (start_frame <= frame_indices) & (frame_indices < end_frame)
            patch_masking_mask = freq_masking_mask.unsqueeze(dim=-1) & time_masking_mask
            specgram = specgram.masked_fill(patch_masking_mask, 0)

            specgram = specgram.view(*batch_shape, n_bins, n_frames)

        sample[output_key] = specgram

        return sample


class NAFPSpectrogramAugmentationComposer(
    NeuralAudioFingerprintingSpectrogramAugmentationComposer
):
    """Alias of NeuralAudioFingerprintingSpectrogramAugmentationComposer."""


def _randrange(
    num_samples: int, param: Tuple[float, float], generator: Optional[torch.Generator] = None
) -> Tuple[int, int]:
    """Generate a random integer in the range [start, end)."""
    min_samples = int(num_samples * param[0])
    max_samples = int(num_samples * param[1]) + 1
    num_masked_samples = torch.randint(min_samples, max_samples, (), generator=generator)
    num_masked_samples = num_masked_samples.item()
    start_index = torch.randint(0, num_samples - num_masked_samples, (), generator=generator)
    start_index = start_index.item()
    end_index = start_index + num_masked_samples

    return start_index, end_index
