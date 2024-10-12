import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config
from audyn.utils.data.mtg_jamendo import download_all_metadata


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    split = config.preprocess.split
    subset = config.preprocess.subset

    assert dump_format == "webdataset", "Only dump_format=webdataset is supported."
    assert list_path is not None, "Specify preprocess.list_path."
    assert split is not None, "Specify preprocess.split."
    assert subset is not None, "Specify preprocess.subset."

    annotations = download_all_metadata(split=split)

    with open(list_path, mode="w") as f:
        for annotation in annotations:
            path = annotation["path"]
            filename = path.replace(".mp3", "")
            path = os.path.join(wav_dir, path)

            if os.path.exists(path):
                f.write(filename + "\n")


if __name__ == "__main__":
    main()
