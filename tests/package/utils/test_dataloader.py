import torch
from torch.utils.data import Dataset

from audyn.utils.data.dataloader import DistributedDataLoader


def test_distributed_dataloader() -> None:
    batch_size = 2
    num_replicas = 2

    dataset = DummyDataset()

    dataloader_rank0 = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=0,
        shuffle=True,
    )
    dataloader_rank1 = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=1,
        shuffle=True,
    )
    data_rank0 = []
    data_rank1 = []

    for data in dataloader_rank0:
        data = data.view(-1).tolist()
        data_rank0 = data_rank0 + data

    for data in dataloader_rank1:
        data = data.view(-1).tolist()
        data_rank1 = data_rank1 + data

    # should be disjoint
    assert set(data_rank0) & set(data_rank1) == set()


class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])

    def __len__(self) -> int:
        return 10
