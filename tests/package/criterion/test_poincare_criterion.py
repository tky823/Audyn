import pytest
import torch

from audyn.criterion.poincare import PoincareDistanceLoss
from audyn.modules import PoincareEmbedding


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_poincare_distance_loss(reduction: str) -> None:
    torch.manual_seed(0)

    batch_shape = (4, 3)
    num_embeddings = 10
    embedding_dim = 2
    curvature = -1

    manifold = PoincareEmbedding(num_embeddings, embedding_dim, curvature=curvature)
    criterion = PoincareDistanceLoss(curvature=curvature, reduction=reduction)

    input = torch.randint(0, num_embeddings, batch_shape, dtype=torch.long)
    target = torch.randint(0, num_embeddings, batch_shape, dtype=torch.long)
    input = manifold(input)
    target = manifold(target)
    loss = criterion(input, target)

    if reduction in ["mean", "sum"]:
        assert loss.size() == ()
    elif reduction == "none":
        assert loss.size() == batch_shape
    else:
        raise ValueError(f"Unexpected {reduction} is found.")
