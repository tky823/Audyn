import os
import shutil
import uuid
import zipfile

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_file


@audyn_main(config_name="download-lsx")
def main(config: DictConfig) -> None:
    """Download LSX dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .zip file.
        lsx_root="${data_root}/LSX"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-lsx \
        root="${data_root}" \
        lsx_root="${lsx_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_lsx(config)


def download_lsx(config: DictConfig) -> None:
    root = config.root
    lsx_root = config.lsx_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://zenodo.org/records/7765140/files/lsx.zip"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    if root:
        os.makedirs(root, exist_ok=True)

    zip_filename = os.path.basename(url)
    zip_path = os.path.join(root, zip_filename)

    if not os.path.exists(zip_path):
        _download_lsx(url, zip_path, chunk_size=chunk_size)

    if unpack:
        if lsx_root is None:
            lsx_root = os.path.join(root, "LSX")

        _unpack_zip(zip_path, lsx_root=lsx_root)


def _download_lsx(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_zip(path: str, lsx_root: str) -> None:
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(lsx_root)
