import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, size: int = 20) -> None:
        super().__init__()

        self.size = size

    def __getitem__(self, idx) -> torch.Tensor:
        value = torch.tensor([idx], dtype=torch.float)
        output = {
            "input": value,
            "target": value,
            "filename": f"utterance-{idx}",
            "subset": f"subset-{idx}",
        }

        return output

    def __len__(self) -> int:
        return self.size


class DummyWaveformDataset(Dataset):
    def __init__(self, size: int = 20, min_length: int = None, max_length: int = None) -> None:
        super().__init__()

        if min_length is None:
            min_length = 1

        if max_length is None:
            raise ValueError("Specify max_length.")

        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        min_length = self.min_length
        max_length = self.max_length

        length = torch.randint(min_length, max_length, ()).item()
        waveform = torch.randn((1, length), dtype=torch.float)
        output = {"input": waveform, "target": waveform}

        return output

    def __len__(self) -> int:
        return self.size


class DummyGANDataset(Dataset):
    def __init__(self, size: int = 20, min_length: int = None, max_length: int = None) -> None:
        super().__init__()

        if min_length is None:
            min_length = 1

        if max_length is None:
            raise ValueError("Specify max_length.")

        self.size = size
        self.min_length = min_length
        self.max_length = max_length

    def __getitem__(self, idx) -> torch.Tensor:
        min_length = self.min_length
        max_length = self.max_length

        length = torch.randint(min_length, max_length, ()).item()
        waveform = torch.randn((1, length), dtype=torch.float)
        fake_label = torch.tensor(0, dtype=torch.float)
        real_label = torch.tensor(1, dtype=torch.float)
        fake_target = torch.tensor(1, dtype=torch.float)

        output = {
            "input": waveform,
            "fake_label": fake_label,
            "real_label": real_label,
            "fake_target": fake_target,
        }

        return output

    def __len__(self) -> int:
        return self.size


class DummySequentialDataset(Dataset):
    def __init__(self, num_features: int, min_length: int, size: int = 20) -> None:
        super().__init__()

        self.num_features = num_features
        self.min_length = min_length
        self.size = size

    def __getitem__(self, idx: int) -> torch.Tensor:
        num_features = self.num_features
        min_length = self.min_length

        shape = (num_features, min_length + idx + 1)
        input = torch.full(shape, fill_value=idx, dtype=torch.float)
        target = torch.tensor(idx, dtype=torch.float)
        output = {
            "input": input,
            "target": target,
        }

        return output

    def __len__(self) -> int:
        return self.size
