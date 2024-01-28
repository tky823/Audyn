import torch

from audyn.transforms.kaldi import KaldiMelSpectrogram


def test_kaldi_melspectrogram() -> None:
    torch.manual_seed(0)

    sample_rate = 16000
    n_mels = 80
    batch_size, in_channels, timesteps = 4, 1, 16000

    # 2D input
    waveform = torch.randn((batch_size, timesteps))

    melspectrogram_transform = KaldiMelSpectrogram(sample_rate, n_mels=n_mels)
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:2] == (batch_size, n_mels)

    # 3D input
    waveform = torch.randn((batch_size, in_channels, timesteps))

    melspectrogram_transform = KaldiMelSpectrogram(sample_rate, n_mels=n_mels)
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:3] == (batch_size, in_channels, n_mels)
