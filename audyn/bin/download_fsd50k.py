import os
import shutil
import subprocess
import sys
import tempfile
import uuid
import warnings
import zipfile
from typing import Optional

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import DEFAULT_CHUNK_SIZE, download_file

IS_WINDOWS = sys.platform == "win32"


@audyn_main(config_name="download-fsd50k")
def main(config: DictConfig) -> None:
    """Download FSD50K dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .zip file.
        fsd50k_root="${data_root}/FSD50K"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-fsd50k \
        root="${data_root}" \
        fsd50k_root="${fsd50k_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_fsd50k(config)


def download_fsd50k(config: DictConfig) -> None:
    root = config.root
    fsd50k_root = config.fsd50k_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://zenodo.org/records/4060432/files/"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if root:
        os.makedirs(root, exist_ok=True)

    for subset in ["dev", "eval"]:
        zip_template = f"FSD50K.{subset}_audio.z" + "{}"

        if subset == "dev":
            num_files = 6
        elif subset == "eval":
            num_files = 2
        else:
            raise ValueError("Invalid subset is given.")

        for idx in range(1, num_files + 1):
            if idx == num_files:
                # "FSD50K.dev_audio.zip"
                filename = zip_template.format("ip")
            else:
                # "FSD50K.dev_audio.z01", ...
                filename = zip_template.format(f"{idx:02d}")

            _url = url + filename
            path = os.path.join(root, filename)

            if not os.path.exists(path):
                _download_fsd50k(_url, path, chunk_size=chunk_size)

    if unpack:
        if IS_WINDOWS:
            warnings.warn("Windows is not fully supported.", UserWarning, stacklevel=2)

        for subset in ["dev", "eval"]:
            zip_template = os.path.join(root, f"FSD50K.{subset}_audio.zip")
            merged_path = zip_template.replace(".zip", ".merged.zip")

            cmd = ["zip", "-s", "0", zip_template, "--out", merged_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            default_name = "FSD50K"

            if fsd50k_root is None:
                fsd50k_root = os.path.join(root, default_name)

            unpacked_root = os.path.join(fsd50k_root)
            _unpack_zip(merged_path, unpacked_root=unpacked_root)


def _download_fsd50k(url: str, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, temp_path, chunk_size=chunk_size)
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
