import glob
import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    fma_root = config.preprocess.fma_root
    subset = config.preprocess.subset

    assert list_path is not None, "Specify preprocess.list_path."
    assert subset is not None, "Specify preprocess.subset."
    assert subset == "evaluation", "'evaluation' is expected as subset."

    subset_dir = os.path.join(
        fma_root,
        "music",
        "test-query-db-500-30s",
        "db",
    )
    paths = sorted(glob.glob(os.path.join(subset_dir, "*", "*.wav")))
    filenames = []

    for path in paths:
        path = os.path.relpath(path, subset_dir)
        filename, _ = os.path.splitext(path)
        filenames.append(filename)

    with open(list_path, mode="w") as f:
        for filename in filenames:
            f.write(f"{filename}\n")


if __name__ == "__main__":
    main()
