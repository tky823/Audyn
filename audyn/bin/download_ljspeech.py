import glob
import os
import shutil
import tarfile
import tempfile
import uuid

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_file


@audyn_main(config_name="download-ljspeech")
def main(config: DictConfig) -> None:
    """Download LJSpeech dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .tar.bz2 file.
        ljspeech_root="${data_root}/LJSpeech-1.1"
        unpack=true  # unpack .tar.bz2 or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-ljspeech \
        root="${data_root}" \
        ljspeech_root="${ljspeech_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_ljspeech(config)


def download_ljspeech(config: DictConfig) -> None:
    root = config.root
    ljspeech_root = config.ljspeech_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    if root:
        os.makedirs(root, exist_ok=True)

    tar_filename = os.path.basename(url)
    tar_path = os.path.join(root, tar_filename)

    if not os.path.exists(tar_path):
        _download_ljspeech(url, tar_path, chunk_size=chunk_size)

    if unpack:
        if ljspeech_root is None:
            ljspeech_root = os.path.join(root, "LJSpeech-1.1")

        _unpack_targz(tar_path, ljspeech_root=ljspeech_root)


def _download_ljspeech(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, temp_path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_targz(path: str, ljspeech_root: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(path) as f:
            f.extractall(temp_dir)

        os.makedirs(ljspeech_root, exist_ok=True)

        for temp_path in glob.glob(os.path.join(temp_dir, "LJSpeech-1.1", "*")):
            shutil.move(temp_path, ljspeech_root)
