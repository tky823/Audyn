import torch

from audyn.transforms.resample import DynamicResample


def test_dynamic_resample() -> None:
    torch.manual_seed(0)

    freq = 16000
    new_freq = 8000

    transform = DynamicResample(sample_rate=new_freq)
    waveform = torch.randn(1, 5 * freq)

    _ = transform(waveform, freq)
