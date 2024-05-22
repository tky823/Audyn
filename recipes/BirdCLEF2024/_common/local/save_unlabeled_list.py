import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.birdclef.birdclef2024 import split


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    audio_root = config.preprocess.audio_root
    subset = config.preprocess.subset
    train_ratio = config.preprocess.train_ratio

    assert list_path is not None, "Specify preprocess.list_path."
    assert audio_root is not None, "Specify preprocess.audio_root."
    assert subset is not None, "Specify preprocess.subset."

    train_filenames, validation_filenames = split(
        audio_root,
        train_ratio=train_ratio,
        seed=config.system.seed,
    )

    if subset == "unlabeled_train":
        filenames = train_filenames
    elif subset == "unlabeled_validation":
        filenames = validation_filenames
    else:
        raise ValueError(f"{subset} is not supported.")

    with open(list_path, mode="w") as f:
        for filename in filenames:
            filename, _ = os.path.splitext(filename)
            f.write(filename + "\n")


if __name__ == "__main__":
    main()
