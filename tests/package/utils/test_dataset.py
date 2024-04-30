import os
import tempfile

import pytest
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from audyn.utils.data import default_collate_fn
from audyn.utils.data.dataset import (
    SortableTorchObjectDataset,
    TorchObjectDataset,
    WebDatasetWrapper,
)

keys = ["input", "scalar"]


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


@pytest.mark.parametrize("sort_key", keys)
def test_sortable_torch_object_dataset(sort_key: str) -> None:
    torch.manual_seed(0)

    list_path = "tests/mock/dataset/torch_object/sample.txt"

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(feature_dir, exist_ok=True)

        with open(list_path) as f:
            for line in f:
                idx = int(line.strip())
                feature_path = os.path.join(feature_dir, f"{idx}.pth")
                scalar = torch.randint(0, 10, (), dtype=torch.long)
                feature = {
                    "input": torch.tensor([idx] * (idx + 1)),
                    "scalar": scalar,
                }

                torch.save(feature, feature_path)

        dataset = SortableTorchObjectDataset(list_path, feature_dir, sort_key=sort_key)
        min_scalar = float("inf")

        for idx, sample in enumerate(dataset):
            idx = len(dataset) - idx

            if sort_key == "input":
                assert torch.equal(torch.tensor([idx] * (idx + 1)), sample[sort_key])
            elif sort_key == "scalar":
                scalar = sample[sort_key].item()

                assert min_scalar >= scalar

                min_scalar = min(min_scalar, scalar)
            else:
                raise ValueError(f"Invalid {sort_key} is detected.")


def test_webdataset_dataset() -> None:
    text_key = "text"
    torchdump_key = "torchdump"
    list_path = "tests/mock/dataset/torch_object/sample.txt"
    batch_size = 2

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        feature_dir = os.path.join(temp_dir, "feature")
        template_path = os.path.join(feature_dir, "%d.tar")

        os.makedirs(feature_dir, exist_ok=True)

        max_shard_size = 5000

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in f:
                idx = int(line.strip())
                feature = {
                    "__key__": str(idx),
                    f"{text_key}.txt": f"{idx}",
                    f"{torchdump_key}.pth": torch.tensor([idx]),
                }

                sink.write(feature)

        dataset = WebDatasetWrapper.instantiate_dataset(list_path, feature_dir)

        for idx, sample in enumerate(dataset):
            assert sample[text_key] == f"{idx + 1}"
            assert torch.equal(torch.tensor([idx + 1]), sample[torchdump_key])

        loader = DataLoader(dataset, batch_size=batch_size, collate_fn=default_collate_fn)

        for idx, batch in enumerate(loader):
            assert [f"{batch_size * idx + 1}", f"{batch_size * (idx + 1)}"] == batch[text_key]
            assert torch.equal(
                torch.tensor([[batch_size * idx + 1], [batch_size * (idx + 1)]]),
                batch[torchdump_key],
            )
