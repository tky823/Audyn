import pytest
import torch
from torch.utils.data import Dataset

from audyn.utils.data.sampler import SequentialBatchSampler


@pytest.mark.parametrize("drop_last", [True, False])
def test_sequential_batch_sampler(drop_last: bool):
    DATA_SIZE = 20
    BATCH_SIZE = 3

    class CustomDataset(Dataset):
        def __init__(self) -> None:
            super().__init__()

        def __getitem__(self, idx) -> torch.Tensor:
            value = torch.tensor([idx], dtype=torch.float)
            output = {"input": value, "target": value}

            return output

        def __len__(self) -> int:
            return DATA_SIZE

    dataset = CustomDataset()
    batch_sampler = SequentialBatchSampler(dataset, batch_size=BATCH_SIZE, drop_last=drop_last)

    indices_set = set()
    indices_list = list()

    for data in batch_sampler:
        indices_set |= set(data)
        indices_list += data

    # confirm no duplicates in indices
    assert len(indices_set) == len(indices_list)
    assert indices_set == set(indices_list)
