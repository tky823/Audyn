import pytest
import torch

from audyn.criterion.lsgan import MSELoss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_lsgan_mse_loss(reduction: str) -> None:
    batch_size = 4
    height, width = 16, 32

    input = torch.rand((batch_size, height, width))

    target = 0
    criterion = MSELoss(target, reduction=reduction)
    loss = criterion(input)

    if reduction == "none":
        assert loss.size() == (batch_size, height, width)
    else:
        assert loss.size() == ()

    target = 1
    criterion = MSELoss(target, reduction=reduction)
    loss = criterion(input)

    if reduction == "none":
        assert loss.size() == (batch_size, height, width)
    else:
        assert loss.size() == ()
