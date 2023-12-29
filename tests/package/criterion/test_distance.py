import torch

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
