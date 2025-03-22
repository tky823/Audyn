import torch

from audyn.criterion.rvqvae import CodebookEntropyLoss, CodebookUsageLoss


def test_rvq_codebook_entropy_loss() -> None:
    torch.manual_seed(0)

    batch_size = 10
    codebook_size = 8
    num_stages = 4

    criterion = CodebookEntropyLoss(codebook_size)
    indices = torch.randint(0, codebook_size, (batch_size, num_stages))
    _ = criterion(indices)


def test_rvq_codebook_usage_loss() -> None:
    torch.manual_seed(0)

    batch_size = 10
    codebook_size = 8
    num_stages = 4

    criterion = CodebookUsageLoss(codebook_size)
    indices = torch.randint(0, codebook_size, (batch_size, num_stages))
    _ = criterion(indices)
