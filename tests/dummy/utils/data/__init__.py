from typing import Dict

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, size: int = 20) -> None:
        super().__init__()

        self.size = size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
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

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
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


class DummyTextToFeatDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int,
        length: int,
        up_scale: int = 2,
        size: int = 20,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.n_mels = n_mels
        self.length = length
        self.up_scale = up_scale
        self.size = size

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        vocab_size = self.vocab_size
        n_mels = self.n_mels
        length = self.length
        up_scale = self.up_scale

        text = torch.randint(0, vocab_size, (length,))
        text_duration = torch.full((length,), fill_value=up_scale, dtype=torch.long)
        melspectrogram = torch.rand((up_scale * length, n_mels))

        output = {
            "text": text,
            "text_duration": text_duration,
            "melspectrogram": melspectrogram,
        }

        return output

    def __len__(self) -> int:
        return self.size


class DummyFeatToWaveDataset(Dataset):
    def __init__(
        self,
        n_mels: int,
        length: int,
        up_scale: int = 2,
        size: int = 20,
    ) -> None:
        super().__init__()

        self.n_mels = n_mels
        self.length = length
        self.up_scale = up_scale
        self.size = size

    def __getitem__(self, _: int) -> Dict[str, torch.Tensor]:
        n_mels = self.n_mels
        length = self.length
        up_scale = self.up_scale

        melspectrogram = torch.randn((n_mels, length))
        waveform = torch.randn((1, up_scale * length))

        output = {
            "melspectrogram": melspectrogram,
            "waveform": waveform,
        }

        return output

    def __len__(self) -> int:
        return self.size


class DummyTextToWaveDataset(Dataset):
    def __init__(
        self,
        vocab_size: int,
        length: int,
        up_scale: int = 2,
        size: int = 20,
    ) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.length = length
        self.up_scale = up_scale
        self.size = size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vocab_size = self.vocab_size
        length = self.length
        up_scale = self.up_scale

        text = torch.randint(0, vocab_size, (length,))
        waveform = torch.randn((1, up_scale * length))

        output = {
            "text": text,
            "waveform": waveform,
            "filename": f"utterance-{idx}",
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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


class DummyVQVAEDataset(Dataset):
    def __init__(
        self,
        num_features: int,
        height: int,
        width: int,
        down_scale: int,
        size: int = 20,
        codebook_size: int = None,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.height, self.width = height, width
        self.down_scale = down_scale
        self.size = size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        num_features = self.num_features
        height, width = self.height, self.width
        down_scale = self.down_scale

        input = torch.randn((num_features, height, width), dtype=torch.float)
        codebook_indices = torch.zeros(
            (height // down_scale, width // down_scale), dtype=torch.long
        )
        output = {
            "input": input,
            "codebook_indices": codebook_indices,
            "filename": f"utterance-{idx}",
            "subset": f"subset-{idx}",
        }

        return output

    def __len__(self) -> int:
        return self.size
