import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["WaveformSlicer"]


class WaveformSlicer(nn.Module):
    """Module to slice waveform by fixed length or duration.

    Args:
        length (int, optional): Length of waveform slice.
        duration (float, optional): Duration of waveform slice.
        sample_rate (int, optional): Sampling rate of waveform.
        seed (int): Random seed.

    .. note::

        Either ``length`` or pair of ``duration`` and ``sample_rate`` is required.

    """

    def __init__(
        self,
        length: int = None,
        duration: float = None,
        sample_rate: int = None,
        seed: int = 0,
    ) -> None:
        super().__init__()

        if length is None:
            assert duration is not None and sample_rate is not None

            length = int(sample_rate * duration)
        else:
            assert duration is None and sample_rate is None

        self.length = length
        self.duration = duration
        self.sample_rate = sample_rate
        self.generator = torch.Generator()

        self.generator.manual_seed(seed)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of WaveformSlicer.

        Args:
            waveform (torch.Tensor): Waveform of shape (\*, length).

        Returns:
            torch.Tensor: Sliced waveform of shape (\*, slice_length).

        .. note::

            If ``self.training=True``, slice is selected at random.
            Otherwise, slice is selected from middle frames.

        """
        length = self.length
        orig_length = waveform.size(-1)
        padding = length - orig_length

        if self.training:
            if padding > 0:
                padding_left = torch.randint(0, padding, (), generator=self.generator)
                padding_left = padding_left.item()
                padding_right = padding - padding_left
                waveform_slice = F.pad(waveform, (padding_left, padding_right))
            elif padding < 0:
                trimming = -padding
                start_idx = torch.randint(0, trimming, (), generator=self.generator)
                padding_left = padding_left.item()
                end_idx = start_idx + length
                _, waveform_slice, _ = torch.split(
                    waveform, [start_idx, length, orig_length - end_idx], dim=-1
                )
            else:
                waveform_slice = waveform
        else:
            if padding > 0:
                padding_left = padding // 2
                padding_right = padding - padding_left
                waveform_slice = F.pad(waveform, (padding_left, padding_right))
            elif padding < 0:
                trimming = -padding
                start_idx = trimming // 2
                end_idx = start_idx + length
                _, waveform_slice, _ = torch.split(
                    waveform, [start_idx, length, orig_length - end_idx], dim=-1
                )
            else:
                waveform_slice = waveform

        return waveform_slice
