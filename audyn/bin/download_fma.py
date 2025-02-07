import glob
import os
import shutil
import tempfile
import uuid
import zipfile

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_file


@audyn_main(config_name="download-fma")
def main(config: DictConfig) -> None:
    """Download FreeMusicArchive (FMA) dataset.

    .. code-block:: shell

        type="medium"  # for FMA-medium

        data_root="./data"  # root directory to save .zip file.
        fma_root="${data_root}/FMA/${type}"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-fma \
        type="${type}" \
        root="${data_root}" \
        fma_root="${fma_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_fma(config)


def download_fma(config: DictConfig) -> None:
    _type = config.type
    root = config.root
    fma_root = config.fma_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    metadata_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    audio_url = f"https://os.unil.cloud.switch.ch/fma/fma_{_type}.zip"

    if root is None:
        raise ValueError("Set root directory.")

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    if root:
        os.makedirs(root, exist_ok=True)

    metadata_filename = os.path.basename(metadata_url)
    metadata_path = os.path.join(root, metadata_filename)

    if not os.path.exists(metadata_path):
        _download_fma(metadata_url, metadata_path, chunk_size=chunk_size)

    audio_filename = os.path.basename(audio_url)
    audio_path = os.path.join(root, audio_filename)

    if not os.path.exists(audio_path):
        _download_fma(audio_url, audio_path, chunk_size=chunk_size)

    if unpack:
        if fma_root is None:
            fma_root = os.path.join(root, "FMA", _type)

        _unpack_zip(metadata_path, fma_root=fma_root)
        _unpack_zip(audio_path, fma_root=fma_root)


def _download_fma(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, temp_path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_zip(path: str, fma_root: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(temp_dir)

        os.makedirs(fma_root, exist_ok=True)

        for temp_path in glob.glob(os.path.join(temp_dir, "*")):
            shutil.move(temp_path, fma_root)
