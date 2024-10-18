from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from ..composer import Composer


class MTGJamendoEvaluationWaveformSliceComposer(Composer):
    """Composer to slice multiple waveforms by fixed length or duration.

    Args:
        input_key (str): Key of tensor to slice waveforms.
        output_key (str): Key of tensor to store sliced waveforms.
        length (int, optional): Length of waveform slice.
        duration (float, optional): Duration of waveform slice.
        sample_rate (int, optional): Sampling rate of waveform.
        num_slices (int): Number of slices.

    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        length: Optional[int] = None,
        duration: Optional[float] = None,
        sample_rate: Optional[int] = None,
        num_slices: int = 1,
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

        self.input_key = input_key
        self.output_key = output_key

        self.length = length
        self.duration = duration
        self.sample_rate = sample_rate
        self.num_slices = num_slices

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key
        length = self.length
        num_slices = self.num_slices

        waveform = sample[input_key]

        *batch_shape, orig_length = waveform.size()
        waveform = waveform.view(-1, 1, 1, orig_length)

        hop_length = (orig_length - length) // (num_slices - 1)
        waveform = F.unfold(
            waveform,
            kernel_size=(1, length),
            stride=(1, hop_length),
        )

        waveform = waveform.permute(1, 0, 2).contiguous()
        waveform = waveform.view(num_slices, *batch_shape, length)

        sample[output_key] = waveform

        keys = set(sample.keys()) - {input_key, output_key}

        for key in keys:
            value = sample[key]
            repeated_value = [value for _ in range(num_slices)]

            if isinstance(value, (str, int, float)):
                # primitive types
                pass
            elif isinstance(value, (list, dict)):
                # structured types
                pass
            elif isinstance(value, torch.Tensor):
                repeated_value = torch.stack(repeated_value, dim=0)
            else:
                raise ValueError(f"{type(value)} is not supported.")

            sample[key] = repeated_value

        return sample
