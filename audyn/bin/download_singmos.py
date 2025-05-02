import csv
import os
import shutil
import uuid
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from ..utils._hydra import main as audyn_main
from ..utils.data.download import DEFAULT_CHUNK_SIZE, download_by_response

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="download-singmos")
def main(config: DictConfig) -> None:
    """Download SingMOS dataset.

    .. code-block:: shell

        data_root="./data"
        singmos_root="${data_root}/SingMOS"
        chunk_size=8192  # chunk size in byte to download

        audyn-download-singmos \
        singmos_root="${singmos_root}" \
        chunk_size=${chunk_size}

    """
    download_singmos(config)


def download_singmos(config: DictConfig) -> None:
    singmos_root = config.singmos_root
    chunk_size = config.chunk_size

    url = "https://huggingface.co/datasets/TangRain/SingMOS/resolve/main"

    if singmos_root is None:
        raise ValueError("Set singmos_root.")

    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    if singmos_root:
        os.makedirs(singmos_root, exist_ok=True)

    filename = "metadata.csv"
    _url = url + f"/{filename}"
    metadata_path = os.path.join(singmos_root, filename)

    if not os.path.exists(metadata_path):
        _download_singmos(_url, metadata_path, chunk_size=chunk_size)

    for filename in ["info/score.json", "info/split.json", "info/sys_info.json"]:
        _url = url + f"/{filename}"
        _path = os.path.join(singmos_root, filename)

        if not os.path.exists(_path):
            info_dir = os.path.dirname(_path)
            os.makedirs(info_dir, exist_ok=True)

            _download_singmos(_url, _path, chunk_size=chunk_size)

    filenames = []

    with open(metadata_path) as f:
        for idx, line in enumerate(csv.reader(f)):
            if idx < 1:
                continue

            _, filename, _, _ = line
            filenames.append(filename)

    if IS_TQDM_AVAILABLE:
        pbar = tqdm(filenames)
    else:
        pbar = filenames

    for filename in pbar:
        _url = url + f"/{filename}"
        wav_path = os.path.join(singmos_root, filename)

        if not os.path.exists(wav_path):
            wav_dir = os.path.dirname(wav_path)
            os.makedirs(wav_dir, exist_ok=True)

            _download_singmos(_url, wav_path, chunk_size=chunk_size)


def _download_singmos(url: str, path: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    request = Request(url)

    try:
        with urlopen(request) as response, open(temp_path, "wb") as f:
            download_by_response(response, f, chunk_size=chunk_size)

        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e
