import glob
import os
import shutil
import tarfile
import tempfile
import uuid

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import download_file


@audyn_main(config_name="download-dnr")
def main(config: DictConfig) -> None:
    """Download DnR dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .tar.gz file.
        dnr_root="${data_root}/DnR-v2"
        version=2
        unpack=true  # unpack .tar.gz or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-dnr \
        root="${data_root}" \
        dnr_root="${dnr_root}" \
        version=${version} \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_dnr(config)


def download_dnr(config: DictConfig) -> None:
    root = config.root
    dnr_root = config.dnr_root
    version = config.version
    unpack = config.unpack
    chunk_size = config.chunk_size

    num_files = 11

    url = "https://zenodo.org/records/6949108/files/"
    targz_template = "dnr_v2.tar.gz.{:02d}"

    if root is None:
        raise ValueError("Set root directory.")

    if version is None:
        version = 2

    if unpack is None:
        unpack = True

    if chunk_size is None:
        chunk_size = 8192

    version = str(version)

    assert version.lower() in ["2", "v2"], "DnR-v2 is supported."

    if root:
        os.makedirs(root, exist_ok=True)

    for idx in range(num_files):
        filename = targz_template.format(idx)
        _url = url + filename
        path = os.path.join(root, filename)

        if not os.path.exists(path):
            _download_dnr(_url, path, chunk_size=chunk_size)

    if unpack:
        merged_path = targz_template.format(0)
        merged_path = merged_path[:-3]

        with open(merged_path, "wb") as f_out:
            for idx in range(num_files):
                filename = targz_template.format(idx)
                path = os.path.join(root, filename)

                with open(path, "rb") as f_in:
                    f_out.write(f_in.read())

        root = os.path.dirname(path)
        default_name = "DnR-v2"

        if dnr_root is None:
            dnr_root = os.path.join(root, default_name)

        unpacked_root = os.path.join(dnr_root, "audio")
        _unpack_targz(merged_path, unpacked_root=unpacked_root)


def _download_dnr(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file(url, path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e


def _unpack_targz(path: str, dnr_root: str) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        with tarfile.open(path) as f:
            f.extractall(temp_dir)

        for temp_path in glob.glob(os.path.join(temp_dir, "*")):
            shutil.move(temp_path, dnr_root)
