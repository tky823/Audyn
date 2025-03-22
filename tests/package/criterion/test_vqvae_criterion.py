import torch

from audyn.criterion.vqvae import CodebookEntropyLoss, CodebookUsageLoss


def test_codebook_entropy_loss() -> None:
    torch.manual_seed(0)

    batch_size = 10
    codebook_size = 8

    criterion = CodebookEntropyLoss(codebook_size)
    indices = torch.randint(0, codebook_size, (batch_size,))
    _ = criterion(indices)


def test_codebook_usage_loss() -> None:
    torch.manual_seed(0)

    batch_size = 10
    codebook_size = 8

    criterion = CodebookUsageLoss(codebook_size)
    indices = torch.randint(0, codebook_size, (batch_size,))
    _ = criterion(indices)
