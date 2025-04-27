import torch

from audyn.criterion.negative_sampling import DistanceBasedNegativeSamplingLoss
from audyn.functional.poincare import poincare_distance


def test_distance_based_negative_sampling_loss() -> None:
    torch.manual_seed(0)

    batch_size = 4
    num_neg_samples = 10
    embedding_dim = 2

    anchor = torch.rand((batch_size, embedding_dim)) - 0.5
    positive = torch.rand((batch_size, embedding_dim)) - 0.5
    negative = torch.rand((batch_size, num_neg_samples, embedding_dim)) - 0.5

    def euclid_distance(input: torch.Tensor, target: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return torch.mean((input - target) ** 2, dim=dim)

    reduction = "mean"
    criterion = DistanceBasedNegativeSamplingLoss(
        euclid_distance,
        reduction=reduction,
        positive_distance_kwargs={"dim": -1},
        negative_distance_kwargs={"dim": -1},
    )
    loss = criterion(anchor, positive, negative)

    assert loss.size() == ()

    reduction = "sum"
    criterion = DistanceBasedNegativeSamplingLoss(poincare_distance, reduction=reduction)
    loss = criterion(anchor, positive, negative)

    assert loss.size() == ()

    reduction = "none"
    criterion = DistanceBasedNegativeSamplingLoss(poincare_distance, reduction=reduction)
    loss = criterion(anchor, positive, negative)

    assert loss.size() == (batch_size,)
