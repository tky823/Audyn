import pytest
import torch

from audyn.transforms.slicer import WaveformSlicer


@pytest.mark.parametrize("slice_length", [1024, 2048, 4096])
def test_waveform_slicer(slice_length: int) -> None:
    torch.manual_seed(0)

    batch_size, in_channels, length = 4, 5, 2049

    slicer = WaveformSlicer(length=slice_length)
    waveform = torch.randn((batch_size, in_channels, length))

    slicer.train()
    sliced_waveform = slicer(waveform)

    assert sliced_waveform.size() == (batch_size, in_channels, slice_length)

    slicer.eval()
    sliced_waveform = slicer(waveform)

    assert sliced_waveform.size() == (batch_size, in_channels, slice_length)
