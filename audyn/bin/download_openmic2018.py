import os
import shutil
import tarfile
import tempfile
import uuid
from typing import Optional
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import DEFAULT_CHUNK_SIZE, download_by_response

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="download-openmic2018")
def main(config: DictConfig) -> None:
    """Download OpenMIC-2018 dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .tgz file.
        openmic2018_root="${data_root}/openmic-2018"
        unpack=true  # unpack .tgz or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-openmic2018 \
        root="${data_root}" \
        openmic2018_root="${openmic2018_root}" \
        unpack=${unpack} \
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
        chunk_size = DEFAULT_CHUNK_SIZE

    if root:
        os.makedirs(root, exist_ok=True)

    filename = os.path.basename(url)
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        _download_openmic2018(url, path, chunk_size=chunk_size)

    if unpack:
        _unpack_tgz(path, openmic2018_root=openmic2018_root)


def _download_openmic2018(url: str, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
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
        openmic2018_root = os.path.join(root, "openmic-2018")

    os.makedirs(openmic2018_root, exist_ok=True)

    with tarfile.open(path, "r:gz") as tar, tempfile.TemporaryDirectory() as temp_dir:
        tar.extractall(temp_dir)
        _openmic2018_root = os.path.join(temp_dir, "openmic-2018")

        for _filename in os.listdir(_openmic2018_root):
            _path = os.path.join(_openmic2018_root, _filename)
            shutil.move(_path, openmic2018_root)


if __name__ == "__main__":
    main()
