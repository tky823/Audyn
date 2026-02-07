import torch

from audyn.transforms.birdclef import BirdCLEF2024BaselineMelSpectrogram


def test_birdclef2024_baseline_melspectrogram() -> None:
    torch.manual_seed(0)

    batch_size = 4
    sample_rate = 32000
    n_mels, n_frames = 128, 384
    duration = 15

    transform = BirdCLEF2024BaselineMelSpectrogram(sample_rate=sample_rate)
    waveform = torch.randn((batch_size, int(sample_rate * duration)))

    melspectrogram = transform(waveform)

    assert melspectrogram.size() == (batch_size, n_mels, n_frames)
