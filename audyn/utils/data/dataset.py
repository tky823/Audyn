import os
from typing import Any, Dict

import torch
from torch.utils.data import Dataset

__all__ = ["TorchObjectDataset", "SortableTorchObjectDataset"]


class TorchObjectDataset(Dataset):
    def __init__(self, list_path: str, feature_dir: str) -> None:
        super().__init__()

        self.feature_dir = feature_dir
        self.filenames = []

        with open(list_path) as f:
            for line in f:
                self.filenames.append(line.strip())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename = self.filenames[idx]
        feature_path = os.path.join(self.feature_dir, f"{filename}.pth")
        data = torch.load(feature_path, map_location=lambda storage, loc: storage)

        return data

    def __len__(self) -> int:
        return len(self.filenames)


class SortableTorchObjectDataset(TorchObjectDataset):
    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        sort_by_length: bool = True,
        sort_key: str = None,
        length_dim: int = -1,
    ) -> None:
        if sort_key is None:
            raise ValueError("Specify sort_key.")

        super().__init__(list_path=list_path, feature_dir=feature_dir)

        if sort_by_length:
            lengths = {}

            for filename in self.filenames:
                feature_path = os.path.join(self.feature_dir, f"{filename}.pth")
                data = torch.load(feature_path, map_location=lambda storage, loc: storage)

                if data[sort_key].dim() == 0:
                    raise NotImplementedError(
                        f"0-dimension tensor (key={sort_key}) is not supported."
                    )
                else:
                    lengths[filename] = data[sort_key].size(length_dim)

            # longest is first
            lengths = sorted(lengths.items(), key=lambda x: x[1], reverse=True)
            self.filenames = [filename for filename, _ in lengths]
