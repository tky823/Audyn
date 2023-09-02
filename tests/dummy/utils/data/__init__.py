import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, size: int = 20) -> None:
        super().__init__()

        self.size = size

    def __getitem__(self, idx) -> torch.Tensor:
        value = torch.tensor([idx], dtype=torch.float)
        output = {"input": value, "target": value}

        return output

    def __len__(self) -> int:
        return self.size
