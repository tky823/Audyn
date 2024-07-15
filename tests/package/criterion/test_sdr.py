import pytest
import torch

from audyn.criterion.sdr import SISDR, NegSISDR, PITNegSISDR


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_sisdr(reduction: str) -> None:
    torch.manual_seed(0)

    batch_size = 4
    length = 128

    criterion = SISDR(reduction=reduction)

    input = torch.randn((batch_size, length))
    target = torch.randn((batch_size, length))
    loss = criterion(input, target)

    if reduction == "none":
        assert loss.size() == (batch_size,)
    else:
        assert loss.size() == ()

    input = torch.randn((length,))
    target = torch.randn((length,))
    loss = criterion(input, target)

    assert loss.size() == ()


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_neg_sisdr(reduction: str) -> None:
    torch.manual_seed(0)

    batch_size = 4
    length = 128

    criterion = NegSISDR(reduction=reduction)

    input = torch.randn((batch_size, length))
    target = torch.randn((batch_size, length))
    loss = criterion(input, target)

    if reduction == "none":
        assert loss.size() == (batch_size,)
    else:
        assert loss.size() == ()

    input = torch.randn((length,))
    target = torch.randn((length,))
    loss = criterion(input, target)

    assert loss.size() == ()


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_pit_neg_sisdr(reduction: str) -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_sources = 3
    length = 128

    criterion = PITNegSISDR(reduction=reduction)

    input = torch.randn((batch_size, num_sources, length))
    target = torch.randn((batch_size, num_sources, length))
    loss = criterion(input, target)

    if reduction == "none":
        assert loss.size() == (batch_size, num_sources)
    else:
        assert loss.size() == ()
