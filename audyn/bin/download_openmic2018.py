import os
import shutil
import tarfile
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from ..utils.data.download import download_by_response
from ..utils.hydra import main as audyn_main

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="download-openmic2018")
def main(config: DictConfig) -> None:
    """Download OpenMIC-2018 dataset.

    .. code-block:: shell

        root="./OpenMIC-2018/raw"  # root directory to store
        chunk_size=8192  # chunk size in byte to download

        audyn-download-openmic2018 \
        root="${root}" \
        chunk_size=${chunk_size}

    """
    download_openmic2018(config)


def download_openmic2018(config: DictConfig) -> None:
    root = config.root
    openmic2018_root = config.openmic2018_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://zenodo.org/records/1432913/files/openmic-2018-v1.0.0.tgz"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    if root:
        os.makedirs(root, exist_ok=True)

    filename = os.path.basename(url)
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        _download_openmic2018(url, path, chunk_size=chunk_size)

    if unpack:
        _unpack_tgz(path, openmic2018_root=openmic2018_root)


def _download_openmic2018(url: str, path: str, chunk_size: int = 8192) -> None:
    filename = os.path.basename(url)
    temp_path = path + str(uuid.uuid4())[:8]

    request = Request(url)

    try:
        with urlopen(request) as response, open(temp_path, "wb") as f:
            if IS_TQDM_AVAILABLE:
                total_size = int(response.headers["Content-Length"])

                with tqdm(unit="B", unit_scale=True, desc=filename, total=total_size) as pbar:
                    download_by_response(response, f, chunk_size=chunk_size, pbar=pbar)
            else:
                download_by_response(response, f, chunk_size=chunk_size)

        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_tgz(path: str, openmic2018_root: Optional[str] = None) -> None:
    root = os.path.dirname(path)

    if openmic2018_root is None:
        filename = os.path.basename(path)
        filename, _ = os.path.splitext(filename)
        openmic2018_root = os.path.join(root, filename)

    os.makedirs(openmic2018_root, exist_ok=True)

    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(openmic2018_root)


if __name__ == "__main__":
    main()
