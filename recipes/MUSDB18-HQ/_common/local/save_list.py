import glob
import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.musdb18 import (
    test_track_names,
    train_track_names,
    validation_track_names,
)


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    musdb18hq_root = config.preprocess.musdb18hq_root
    subset = config.preprocess.subset

    assert dump_format == "musdb18", "Only dump_format=musdb18 is supported."
    assert list_path is not None, "Specify preprocess.list_path."
    assert subset is not None, "Specify preprocess.subset."

    if subset in ["train", "validation"]:
        subset_name = "train"
    elif subset == "test":
        subset_name = "test"
    else:
        raise ValueError(f"{subset} is not supported as subset.")

    paths = sorted(glob.glob(os.path.join(musdb18hq_root, subset_name, "*")))
    paths = [path for path in paths if is_included(path, subset=subset)]

    with open(list_path, mode="w") as f:
        for path in paths:
            filename = os.path.basename(path)
            filename = filename.replace(".stem.mp4", "")
            f.write(filename + "\n")


def is_included(path: str, subset: str) -> bool:
    filename = os.path.basename(path)

    if filename.endswith(".stem.mp4"):
        return False

    if subset == "train":
        if filename in train_track_names:
            return True
        else:
            return False
    elif subset == "validation":
        if filename in validation_track_names:
            return True
        else:
            return False
    elif subset == "test":
        if filename in test_track_names:
            return True
        else:
            return False

    return False


if __name__ == "__main__":
    main()
