import glob
import os

import torch
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import setup_config
from audyn.utils.data.fma import (
    full_test_track_ids,
    full_train_track_ids,
    full_validation_track_ids,
    large_test_track_ids,
    large_train_track_ids,
    large_validation_track_ids,
    medium_test_track_ids,
    medium_train_track_ids,
    medium_validation_track_ids,
    small_test_track_ids,
    small_train_track_ids,
    small_validation_track_ids,
)


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    fma_root = config.preprocess.fma_root
    _type = config.preprocess.type
    subset = config.preprocess.subset
    num_samples = config.preprocess.num_samples

    assert list_path is not None, "Specify preprocess.list_path."
    assert _type is not None, "Specify preprocess.type."
    assert subset is not None, "Specify preprocess.subset."
    assert num_samples is not None, "Specify preprocess.num_samples."

    track_ids = []

    for idx in tqdm(range(156)):
        name = f"{idx:03d}"
        template_path = os.path.join(fma_root, "audio", name, "*.mp3")
        paths = sorted(glob.glob(template_path))

        for path in paths:
            if is_included(path, type=_type, subset=subset):
                _, ext = os.path.splitext(path)

                assert ext == ".mp3"

                ext = ext[1:]

                try:
                    torchaudio.load(path, format=ext)
                except RuntimeError:
                    continue

                filename = os.path.basename(path)
                filename, _ = os.path.splitext(filename)
                track_id = int(filename)
                track_ids.append(track_id)

    track_ids = sorted(track_ids)

    g = torch.Generator()
    g.manual_seed(config.system.seed)

    indices = torch.randperm(len(track_ids), generator=g)
    indices = indices.tolist()
    indices = indices[:num_samples]
    indices = sorted(indices)

    with open(list_path, mode="w") as f:
        for index in indices:
            track_id = track_ids[index]
            f.write(f"{track_id}\n")


def is_included(path: str, type: str, subset: str) -> bool:
    if type == "small":
        train_track_ids = small_train_track_ids
        validation_track_ids = small_validation_track_ids
        test_track_ids = small_test_track_ids
    elif type == "medium":
        train_track_ids = medium_train_track_ids
        validation_track_ids = medium_validation_track_ids
        test_track_ids = medium_test_track_ids
    elif type == "large":
        train_track_ids = large_train_track_ids
        validation_track_ids = large_validation_track_ids
        test_track_ids = large_test_track_ids
    elif type == "full":
        train_track_ids = full_train_track_ids
        validation_track_ids = full_validation_track_ids
        test_track_ids = full_test_track_ids
    else:
        raise ValueError(f"Invalid type {type} is found.")

    filename = os.path.basename(path)
    filename, _ = os.path.splitext(filename)
    track_id = int(filename)

    if subset == "train":
        if track_id in train_track_ids:
            return True
        else:
            return False
    elif subset == "validation":
        if track_id in validation_track_ids:
            return True
        else:
            return False
    elif subset == "test":
        if track_id in test_track_ids:
            return True
        else:
            return False

    return False


if __name__ == "__main__":
    main()
