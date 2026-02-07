import pytest
import torch

from audyn.criterion.melspectrogram import MelSpectrogramL1Loss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_melspectrogram_l1_loss(reduction: str) -> None:
    torch.manual_seed(0)

    batch_size, length = 4, 16000

    criterion = MelSpectrogramL1Loss(reduction=reduction)
    input = torch.randn((batch_size, length))
    target = torch.randn((batch_size, length))

    loss = criterion(input, target)

    if reduction == "none":
        assert loss.dim() == 3
    else:
        assert loss.size() == ()
