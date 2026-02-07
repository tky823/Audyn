import pytest
import torch

from audyn.criterion.ssast import ClassificationLoss, ReconstructionLoss


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_ssast_reconstruction_loss(reduction: str) -> None:
    torch.manual_seed(0)

    embedding_dim = 8
    max_length = 30
    batch_size = 4

    reconstruction_criterion = ReconstructionLoss(reduction=reduction)
    classification_criterion = ClassificationLoss(reduction=reduction)

    reconstruction_output = torch.randn((batch_size, max_length, embedding_dim))
    reconstruction_target = torch.randn((batch_size, max_length, embedding_dim))
    classification_output = torch.randn((batch_size, max_length, embedding_dim))
    classification_target = torch.randn((batch_size, max_length, embedding_dim))
    reconstruction_length = torch.randint(
        max_length // 2, max_length, (batch_size,), dtype=torch.long
    )
    classification_length = torch.randint(
        max_length // 2, max_length, (batch_size,), dtype=torch.long
    )

    reconstruction_loss = reconstruction_criterion(
        reconstruction_output, reconstruction_target, length=reconstruction_length
    )
    classification_loss = classification_criterion(
        classification_output, classification_target, length=classification_length
    )

    if reduction in ["mean", "sum"]:
        assert reconstruction_loss.size() == ()
        assert classification_loss.size() == ()
    else:
        assert reconstruction_loss.size() == (batch_size, max_length, embedding_dim)
        assert classification_loss.size() == (batch_size, max_length)
