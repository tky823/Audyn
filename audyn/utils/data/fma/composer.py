from typing import Any, Dict, Optional

import torch

from ..composer import Composer

__all__ = [
    "NAFPWaveformSliceComposer",
]


class NAFPWaveformSliceComposer(Composer):
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
