import os
import shutil
import tempfile
import uuid
import zipfile

import torch
from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import DEFAULT_CHUNK_SIZE, download_file


@audyn_main(config_name="download-beatthis")
def main(config: DictConfig) -> None:
    """Download datasets used for beatthis.

    .. code-block:: shell

        data_root="./data"  # root directory to save .tar.gz file.
        unpack=true  # unpack .tar.gz or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-beatthis \
        root="${data_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_beatthis(config)


def download_beatthis(config: DictConfig) -> None:
    root = config.root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://zenodo.org/records/13922116/files/"
    filenames = [
        "asap",
        "ballroom",
        "beatles",
        "candombe",
        "filosax",
        "groove_midi",
        "gtzan",
        "guitarset",
        "hainsworth",
        "harmonix",
        "hjdb",
        "jaah",
        "rwc",
        "simac",
        "smc",
        "tapcorrect",
    ]

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if root:
        os.makedirs(root, exist_ok=True)

    for filename in filenames:
        _url = url + f"{filename}.zip"
        path = os.path.join(root, f"{filename}.zip")

        if not os.path.exists(path):
            _download_file(_url, path, chunk_size=chunk_size)

    if unpack:
        for filename in filenames:
            _url = url + f"{filename}.zip"
            path = os.path.join(root, f"{filename}.zip")
            _root = os.path.join(root, filename)
            _unpack_zip(path, f"{filename}.npz", _root)


def _download_file(url: str, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, temp_path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_zip(path: str, filename: str, root: str) -> None:
    import numpy as np

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(temp_dir)

        temp_path = os.path.join(temp_dir, filename)

        data = np.load(temp_path)

        for _filename, feature in data.items():
            subset = os.path.dirname(_filename)
            _root = os.path.join(root, subset)
            _path = os.path.join(root, f"{_filename}.pth")

            if os.path.exists(_path):
                continue

            os.makedirs(_root, exist_ok=True)

            feature = torch.from_numpy(feature)
            feature = feature.transpose(0, 1)
            torch.save(feature, _path)
