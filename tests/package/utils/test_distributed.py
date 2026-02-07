import pytest
import torch
from torch.utils.data import Dataset

from audyn.utils.data.distributed import (
    DistributedDynamicBatchSampler,
    DistributedSequentialBatchSampler,
)


@pytest.mark.parametrize("drop_last", [True, False])
def test_distributed_sequential_batch_sampler(drop_last: bool):
    NUM_REPLICAS = 4
    DATA_SIZE = 20
    BATCH_SIZE = 4

    class CustomDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()

            self.data_source = [[idx] * idx for idx in range(1, DATA_SIZE + 1)]

        def __getitem__(self, idx) -> torch.Tensor:
            value = torch.tensor(self.data_source[idx], dtype=torch.float)
            output = {"input": value, "target": value}

            return output

        def __len__(self) -> int:
            return DATA_SIZE

    dataset = CustomDataset()

    total_indices_set = set()
    total_indices_list = list()

    for rank in range(NUM_REPLICAS):
        sampler = DistributedSequentialBatchSampler(
            dataset,
            batch_size=BATCH_SIZE,
            num_replicas=NUM_REPLICAS,
            rank=rank,
            shuffle=drop_last,
            seed=0,
            drop_last=drop_last,
        )

        indices_set = set()
        indices_list = list()

        for data in sampler:
            indices_set |= set(data)
            indices_list += data

        if drop_last:
            # confirm no duplicates among iterations
            assert len(indices_set) == len(indices_list)

        assert indices_set == set(indices_list)

        total_indices_set |= indices_set
        total_indices_list += indices_list

    if drop_last:
        # confirm no duplicates among replicas
        assert len(total_indices_set) == len(total_indices_list)

    assert total_indices_set == set(total_indices_list)


@pytest.mark.parametrize("drop_last", [True, False])
def test_distributed_dynamic_batch_sampler(drop_last: bool):
    NUM_REPLICAS = 4
    DATA_SIZE = 20
    BATCH_LENGTH = 5

    class CustomDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()

            self.data_source = [[idx] * idx for idx in range(1, DATA_SIZE + 1)]

        def __getitem__(self, idx) -> torch.Tensor:
            value = torch.tensor(self.data_source[idx], dtype=torch.float)
            output = {"input": value, "target": value}

            return output

        def __len__(self) -> int:
            return DATA_SIZE

    dataset = CustomDataset()

    total_indices_set = set()
    total_indices_list = list()

    for rank in range(NUM_REPLICAS):
        sampler = DistributedDynamicBatchSampler(
            dataset,
            key="input",
            batch_length=BATCH_LENGTH,
            num_replicas=NUM_REPLICAS,
            rank=rank,
            shuffle=drop_last,
            seed=0,
            drop_last=drop_last,
        )

        indices_set = set()
        indices_list = list()

        for data in sampler:
            indices_set |= set(data)
            indices_list += data

        if drop_last:
            # confirm no duplicates among iterations
            assert len(indices_set) == len(indices_list)

        assert indices_set == set(indices_list)

        total_indices_set |= indices_set
        total_indices_list += indices_list

    if drop_last:
        # confirm no duplicates among replicas
        assert len(total_indices_set) == len(total_indices_list)

    assert total_indices_set == set(total_indices_list)
