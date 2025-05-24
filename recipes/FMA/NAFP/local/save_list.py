import glob
import os

from omegaconf import DictConfig

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

    assert list_path is not None, "Specify preprocess.list_path."
    assert _type is not None, "Specify preprocess.type."
    assert subset is not None, "Specify preprocess.subset."

    track_ids = []

    for idx in range(156):
        name = f"{idx:03d}"
        template_path = os.path.join(fma_root, "audio", name, "*.mp3")
        paths = sorted(glob.glob(template_path))

        for path in paths:
            if is_included(path, type=_type, subset=subset):
                filename = os.path.basename(path)
                filename, _ = os.path.splitext(filename)
                track_id = int(filename)
                track_ids.append(track_id)

    track_ids = sorted(track_ids)

    with open(list_path, mode="w") as f:
        for track_id in track_ids:
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
