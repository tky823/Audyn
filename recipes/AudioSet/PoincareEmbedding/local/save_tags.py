import os

from omegaconf import DictConfig
from utils import tags

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    preprocess_config = config.preprocess
    list_path = preprocess_config.list_path

    list_dir = os.path.dirname(list_path)

    if list_dir:
        os.makedirs(list_dir, exist_ok=True)

    with open(list_path, mode="w") as f:
        for tag in tags:
            line = tag + "\n"
            f.write(line)


if __name__ == "__main__":
    main()
