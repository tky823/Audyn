import glob
import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    list_path = config.preprocess.list_path
    audio_root = config.preprocess.audio_root
    subset = config.preprocess.subset

    assert list_path is not None, "Specify preprocess.list_path."
    assert audio_root is not None, "Specify preprocess.audio_root."
    assert subset is not None, "Specify preprocess.subset."
    assert subset == "test", "Only test is supported as subset."

    paths = sorted(glob.glob(os.path.join(audio_root, "*.ogg")))

    with open(list_path, mode="w") as f:
        for path in paths:
            path = os.path.relpath(path, audio_root)
            filename, _ = os.path.splitext(path)
            f.write(filename + "\n")


if __name__ == "__main__":
    main()
