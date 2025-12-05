from typing import Any, Dict

import torch
import torchaudio
from torch.utils.data import Dataset


class FreeMusicArchiveNAFPDataset(Dataset):
    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        background_list_path: str = None,
        impulse_response_list_path: str = None,
        waveform_key: str = "waveform",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        seed: int = 0,
    ) -> None:
        super().__init__()

        self.feature_dir = feature_dir

        self.filenames = []
        self.background_filenames = []
        self.impulse_response_filenames = []

        with open(list_path) as f:
            for line in f:
                self.filenames.append(line.strip("\n"))

        with open(background_list_path) as f:
            for line in f:
                self.background_filenames.append(line.strip("\n"))

        with open(impulse_response_list_path) as f:
            for line in f:
                self.impulse_response_filenames.append(line.strip("\n"))

        self.waveform_key = waveform_key
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key

        self.seed = seed
        self.generator = None

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        waveform_key = self.waveform_key
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        filename = self.filenames[idx]
        feature = {}

        if self.generator is None:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed)

        waveform, sample_rate = torchaudio.load(filename)

        feature = {
            waveform_key: waveform,
            sample_rate_key: torch.tensor(sample_rate, dtype=torch.long),
            filename_key: filename,
        }

        return feature

    def __len__(self) -> int:
        return len(self.filenames)
