import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.birdclef.birdclef2024 import stratified_split


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    csv_path = config.preprocess.csv_path
    subset = config.preprocess.subset
    train_ratio = config.preprocess.train_ratio

    assert list_path is not None, "Specify preprocess.list_path."
    assert csv_path is not None, "Specify preprocess.csv_path."
    assert subset is not None, "Specify preprocess.subset."
    assert train_ratio is not None, "Specify preprocess.train_ratio."

    train_filenames, validation_filenames = stratified_split(
        csv_path,
        train_ratio=train_ratio,
        seed=config.system.seed,
    )

    if subset == "train":
        filenames = train_filenames
    elif subset == "validation":
        filenames = validation_filenames
    else:
        raise ValueError(f"{subset} is not supported.")

    with open(list_path, mode="w") as f:
        for filename in filenames:
            filename, _ = os.path.splitext(filename)
            f.write(filename + "\n")


if __name__ == "__main__":
    main()
