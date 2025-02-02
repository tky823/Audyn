import torch

from audyn.criterion.bsrnn import SpectrogramL1SNR, WaveformL1SNR


def test_spectrogram_l1snr() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    n_bins, n_frames = 33, 64

    shape = (batch_size, in_channels, n_bins, n_frames)

    criterion = SpectrogramL1SNR()
    input = torch.randn(shape) + 1j * torch.randn(shape)
    target = torch.randn(shape) + 1j * torch.randn(shape)
    output = criterion(input, target)

    assert output.size() == ()


def test_waveform_l1snr() -> None:
    torch.manual_seed(0)

    batch_size = 4
    in_channels = 1
    n_fft, hop_length = 64, 16
    n_bins, n_frames = (n_fft // 2 + 1), 64

    shape = (batch_size, in_channels, n_bins, n_frames)
    window = torch.hann_window(n_fft)

    criterion = WaveformL1SNR(n_fft=n_fft, hop_length=hop_length, window=window)
    input = torch.randn(shape) + 1j * torch.randn(shape)
    target = torch.randn(shape) + 1j * torch.randn(shape)
    output = criterion(input, target)

    assert output.size() == ()
