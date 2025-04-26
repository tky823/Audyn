import glob
import os
import shutil
import tempfile
import uuid
import zipfile

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_file


@audyn_main(config_name="download-vctk")
def main(config: DictConfig) -> None:
    """Download VCTK dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .zip file.
        vctk_root="${data_root}/VCTK"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-vctk \
        root="${data_root}" \
        vctk_root="${vctk_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_vctk(config)


def download_vctk(config: DictConfig) -> None:
    root = config.root
    vctk_root = config.vctk_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"

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
        _download_vctk(url, tar_path, chunk_size=chunk_size)

    if unpack:
        if vctk_root is None:
            vctk_root = os.path.join(root, "VCTK")

        _unpack_zip(tar_path, vctk_root=vctk_root)


def _download_vctk(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, temp_path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_zip(path: str, vctk_root: str) -> None:
    root = os.path.dirname(path)

    if vctk_root is None:
        filename = os.path.basename(path)
        filename, _ = os.path.splitext(filename)
        vctk_root = os.path.join(root, filename)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path) as f:
            f.extractall(temp_dir)

        os.makedirs(vctk_root, exist_ok=True)

        for temp_path in glob.glob(os.path.join(temp_dir, "VCTK-Corpus-0.92", "*")):
            shutil.move(temp_path, vctk_root)
