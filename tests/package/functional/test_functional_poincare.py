import torch

from audyn.functional.poincare import poincare_distance


def test_poincare_distance() -> None:
    torch.manual_seed(0)

    batch_shape = (4, 3)
    embedding_dim = 2

    input = torch.randn((*batch_shape, embedding_dim)) - 0.5
    other = torch.rand((*batch_shape, embedding_dim)) - 0.5

    loss = poincare_distance(input, other, dim=-1)

    assert loss.size() == batch_shape
