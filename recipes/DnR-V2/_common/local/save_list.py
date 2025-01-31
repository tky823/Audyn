import glob
import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.dnr import (
    v2_test_track_names,
    v2_train_track_names,
    v2_validation_track_names,
)


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    dnr_root = config.preprocess.dnr_root
    subset = config.preprocess.subset

    assert dump_format == "dnr-v2", "Only dump_format=dnr-v2 is supported."
    assert list_path is not None, "Specify preprocess.list_path."
    assert subset is not None, "Specify preprocess.subset."

    if subset == "train":
        subset_name = "tr"
    elif subset == "validate":
        subset_name = "cv"
    elif subset == "evaluate":
        subset_name = "tt"
    else:
        raise ValueError(f"{subset} is not supported as subset.")

    paths = sorted(glob.glob(os.path.join(dnr_root, subset_name, "*")))
    filenames = [os.path.basename(path) for path in paths if is_included(path, subset=subset)]
    filenames = [int(filename) for filename in filenames]
    filenames = sorted(filenames)

    with open(list_path, mode="w") as f:
        for filename in filenames:
            f.write(f"{filename}\n")


def is_included(path: str, subset: str) -> bool:
    filename = os.path.basename(path)

    if "." in filename:
        return False

    filename = int(filename)

    if subset == "train":
        if filename in v2_train_track_names:
            return True
        else:
            return False
    elif subset == "validate":
        if filename in v2_validation_track_names:
            return True
        else:
            return False
    elif subset == "evaluate":
        if filename in v2_test_track_names:
            return True
        else:
            return False

    return False


if __name__ == "__main__":
    main()
