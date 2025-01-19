import os
import shutil
import tempfile
import uuid
import zipfile
from typing import Optional
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_by_response

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="download-mtat")
def main(config: DictConfig) -> None:
    """Download MagnaTagATune (MTAT) dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .zip file.
        mtat_root="${data_root}/MTAT"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-mtat \
        root="${data_root}" \
        mtat_root="${mtat_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_mtat(config)


def download_mtat(config: DictConfig) -> None:
    root = config.root
    mtat_root = config.mtat_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    num_files = 3

    url = "https://mirg.city.ac.uk/datasets/magnatagatune/"
    zip_template = "mp3.zip.{:03d}"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    if root:
        os.makedirs(root, exist_ok=True)

    # annotation
    _url = "https://github.com/tky823/Audyn/releases/download/v0.0.4/annotation.zip"
    filename = os.path.basename(_url)
    annotation_path = os.path.join(root, filename)

    if not os.path.exists(annotation_path):
        _download_mtat(_url, annotation_path, chunk_size=chunk_size)

    # audio
    for idx in range(num_files):
        filename = zip_template.format(idx + 1)
        _url = url + filename
        path = os.path.join(root, filename)

        if not os.path.exists(path):
            _download_mtat(_url, path, chunk_size=chunk_size)

    if unpack:
        merged_path = zip_template.format(0)
        merged_path = merged_path[:-4]

        with open(merged_path, "wb") as f_out:
            for idx in range(num_files):
                filename = zip_template.format(idx + 1)
                path = os.path.join(root, filename)

                with open(path, "rb") as f_in:
                    f_out.write(f_in.read())

        root = os.path.dirname(path)
        default_name = "MTAT"

        if mtat_root is None:
            mtat_root = os.path.join(root, default_name)

        _unpack_zip(annotation_path, unpacked_root=mtat_root)

        unpacked_root = os.path.join(mtat_root, "audio")
        _unpack_zip(merged_path, unpacked_root=unpacked_root)


def _download_mtat(url: str, path: str, chunk_size: int = 8192) -> None:
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


def _unpack_zip(path: str, unpacked_root: Optional[str] = None) -> None:
    os.makedirs(unpacked_root, exist_ok=True)

    with zipfile.ZipFile(path) as f, tempfile.TemporaryDirectory() as temp_dir:
        f.extractall(temp_dir)

        for _filename in os.listdir(temp_dir):
            _path = os.path.join(temp_dir, _filename)
            shutil.move(_path, unpacked_root)
