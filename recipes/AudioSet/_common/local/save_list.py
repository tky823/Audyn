import glob
import os

from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    audio_format = config.preprocess.audio_format
    download_dir = config.preprocess.download_dir
    list_path = config.preprocess.list_path
    subset = config.preprocess.subset

    assert list_path is not None, "Specify preprocess.list_path."

    template_path = os.path.join(download_dir, "**", f"*.{audio_format}")
    audio_paths = glob.glob(template_path, recursive=True)
    audio_paths = sorted(audio_paths)

    with open(list_path, "w") as f:
        for audio_path in audio_paths:
            _audio_path = os.path.relpath(audio_path, download_dir)
            _audio_path = os.path.join(subset, _audio_path)
            _audio_path, _ = os.path.splitext(_audio_path)
            line = _audio_path + "\n"
            f.write(line)


if __name__ == "__main__":
    main()
