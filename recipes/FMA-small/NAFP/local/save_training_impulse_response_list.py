import glob
import os

import torch
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
    assert subset in ["train", "validation"], "'train' or 'validation' is expected as subset."

    subset_dir = os.path.join(fma_root, "aug", "ir", "tr")
    paths = sorted(glob.glob(os.path.join(subset_dir, "**", "*.wav"), recursive=True))
    filenames = []

    for path in paths:
        path = os.path.relpath(path, subset_dir)
        filename, _ = os.path.splitext(path)
        filenames.append(filename)

    g = torch.Generator()
    g.manual_seed(config.system.seed)

    indices = torch.randperm(len(filenames), generator=g)
    indices = indices.tolist()

    with open(list_path, mode="w") as f:
        for index in indices:
            filename = filenames[index]
            f.write(f"{filename}\n")


if __name__ == "__main__":
    main()
