import torch
import torch.nn as nn
import torchaudio.transforms as aT

from audyn.criterion.distance import MultiScaleSpectralLoss


def test_multi_scale_spectral_loss() -> None:
    torch.manual_seed(0)

    n_fft = [2**idx for idx in range(6, 12)]
    batch_size, in_channels, length = 2, 3, 8000

    criterion = MultiScaleSpectralLoss(n_fft)

    # 2D input
    input = torch.randn((batch_size, length), dtype=torch.float)
    target = torch.randn((batch_size, length), dtype=torch.float)

    _ = criterion(input, target)

    # 3D input
    input = torch.randn((batch_size, in_channels, length), dtype=torch.float)
    target = torch.randn((batch_size, in_channels, length), dtype=torch.float)

    _ = criterion(input, target)

    # specify transform
    sample_rate = 8000
    n_mels = 16

    transform = []

    for _n_fft in n_fft:
        transform.append(
            aT.MelSpectrogram(sample_rate, n_mels=n_mels, n_fft=_n_fft, hop_length=_n_fft // 4)
        )

    transform = nn.ModuleList(transform)
    criterion = MultiScaleSpectralLoss(n_fft, transform=transform)

    input = torch.randn((batch_size, length), dtype=torch.float)
    target = torch.randn((batch_size, length), dtype=torch.float)

    _ = criterion(input, target)
