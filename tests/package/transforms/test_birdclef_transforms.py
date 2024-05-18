import torch

from audyn.transforms.birdclef import BirdCLEF2024BaselineMelSpectrogram


def test_birdclef2024_baseline_melspectrogram() -> None:
    torch.manual_seed(0)

    sample_rate = 32000
    duration = 15

    transform = BirdCLEF2024BaselineMelSpectrogram(sample_rate=sample_rate)
    waveform = torch.randn((duration,))

    melspectrogram = transform(waveform)

    print(melspectrogram.size())
