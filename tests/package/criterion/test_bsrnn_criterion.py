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
    n_frames = 128

    shape = (batch_size, in_channels, n_frames)

    criterion = WaveformL1SNR()
    input = torch.randn(shape)
    target = torch.randn(shape)
    output = criterion(input, target)

    assert output.size() == ()
