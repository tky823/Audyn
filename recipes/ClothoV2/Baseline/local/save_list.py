import csv
import os

from omegaconf import DictConfig

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    captions_path = config.preprocess.captions_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert captions_path is not None, "Specify preprocess.captions_path."

    filenames = []

    with open(captions_path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            filename, *_ = line
            filename, _ = os.path.splitext(filename)
            filenames.append(filename)

    list_dir = os.path.dirname(list_path)

    os.makedirs(list_dir, exist_ok=True)

    with open(list_path, mode="w") as f:
        for filename in filenames:
            filename = filename.strip()
            f.write(filename + "\n")


if __name__ == "__main__":
    main()
