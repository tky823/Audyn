import pytest
import torch
from torch.utils.data import Dataset

from audyn.utils.data.sampler import DynamicBatchSampler, SequentialBatchSampler


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


@pytest.mark.parametrize("channels_last", [True, False])
def test_dynamic_batch_sampler(channels_last: bool) -> None:
    torch.manual_seed(0)

    DATA_SIZE = 10
    NUM_CHENNELS = 2

    class CustomDataset(Dataset):
        def __init__(self, channels_last: bool) -> None:
            super().__init__()

            self.channels_last = channels_last

        def __getitem__(self, idx: int) -> torch.Tensor:
            channels_last = self.channels_last

            if channels_last:
                input = torch.randn(((idx + 1), NUM_CHENNELS))
            else:
                input = torch.randn((NUM_CHENNELS, (idx + 1)))

            target = torch.tensor([idx], dtype=torch.long)

            output = {"input": input, "target": target}

            return output

        def __len__(self) -> int:
            return DATA_SIZE

    if channels_last:
        dim = -2
    else:
        dim = -1

    dataset = CustomDataset(channels_last)
    batch_sampler = DynamicBatchSampler(
        dataset,
        key="input",
        batch_length=3 * DATA_SIZE // 2,
        dim=dim,
        shuffle=False,
    )
    expected_batch_indices = [[0, 1, 2, 3, 4], [5, 6, 7], [8, 9]]

    assert len(batch_sampler) == len(expected_batch_indices)

    for batch_indices, _expected_batch_indices in zip(batch_sampler, expected_batch_indices):
        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        _expected_batch_indices = torch.tensor(_expected_batch_indices, dtype=torch.long)

        assert torch.equal(batch_indices, _expected_batch_indices)
