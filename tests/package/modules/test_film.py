import torch

from audyn.modules.film import FiLM, FiLM1d, FiLM2d


def test_film1d() -> None:
    torch.manual_seed(0)

    batch_size, in_channels = 4, 3
    length = 5

    input = torch.randn((batch_size, in_channels, length))
    gamma = torch.randn((batch_size, in_channels))
    beta = torch.randn((batch_size, in_channels))

    model = FiLM1d()
    output = model(input, gamma, beta)

    assert output.size() == input.size()


def test_film2d() -> None:
    torch.manual_seed(0)

    batch_size, in_channels = 4, 3
    height, width = 5, 6

    input = torch.randn((batch_size, in_channels, height, width))
    gamma = torch.randn((batch_size, in_channels))
    beta = torch.randn((batch_size, in_channels))

    model = FiLM2d()
    output = model(input, gamma, beta)

    assert output.size() == input.size()


def test_film() -> None:
    torch.manual_seed(0)

    batch_size, in_channels = 4, 3

    length = 5
    input = torch.randn((batch_size, in_channels, length), dtype=torch.float)
    gamma = torch.randn((batch_size, in_channels))
    beta = torch.randn((batch_size, in_channels))

    model = FiLM()
    output = model(input, gamma, beta)

    assert output.size() == input.size()

    height, width, depth = 5, 6, 7

    input = torch.randn((batch_size, in_channels, height, width, depth))
    gamma = torch.randn((batch_size, in_channels))
    beta = torch.randn((batch_size, in_channels))

    model = FiLM()
    output = model(input, gamma, beta)

    assert output.size() == input.size()
