import os
import tempfile

import torch

from audyn.utils.data.dataset import SortableTorchObjectDataset, TorchObjectDataset


def test_torch_object_dataset() -> None:
    key = "input"
    list_path = "tests/mock/dataset/torch_object/sample.txt"

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(feature_dir, exist_ok=True)

        with open(list_path) as f:
            for line in f:
                idx = int(line.strip())
                feature_path = os.path.join(feature_dir, f"{idx}.pth")
                feature = {key: torch.tensor([idx])}

                torch.save(feature, feature_path)

        dataset = TorchObjectDataset(list_path, feature_dir)

        for idx, sample in enumerate(dataset):
            assert torch.equal(torch.tensor([idx + 1]), sample[key])


def test_sortable_torch_object_dataset() -> None:
    key = "input"
    list_path = "tests/mock/dataset/torch_object/sample.txt"

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(feature_dir, exist_ok=True)

        with open(list_path) as f:
            for line in f:
                idx = int(line.strip())
                feature_path = os.path.join(feature_dir, f"{idx}.pth")
                feature = {key: torch.tensor([idx] * (idx + 1))}

                torch.save(feature, feature_path)

        dataset = SortableTorchObjectDataset(list_path, feature_dir, sort_key=key)

        for idx, sample in enumerate(dataset):
            idx = len(dataset) - idx
            assert torch.equal(torch.tensor([idx] * (idx + 1)), sample[key])
