import os
import shutil
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


@audyn_main(config_name="download-song-describer")
def main(config: DictConfig) -> None:
    """Download SongDescriber dataset.

    .. code-block:: shell

        data_root="./data"  # root directory to save .zip file.
        song_describer_root="${data_root}/SongDescriber"
        unpack=true  # unpack .zip or not
        chunk_size=8192  # chunk size in byte to download

        audyn-download-song-describer \
        root="${data_root}" \
        song_describer_root="${song_describer_root}" \
        unpack=${unpack} \
        chunk_size=${chunk_size}

    """
    download_song_describer(config)


def download_song_describer(config: DictConfig) -> None:
    root = config.root
    song_describer_root = config.song_describer_root
    unpack = config.unpack
    chunk_size = config.chunk_size

    url = "https://zenodo.org/records/10072001/files/audio.zip"

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
        _download_song_describer(url, path, chunk_size=chunk_size)

    if unpack:
        _unpack_zip(path, song_describer_root=song_describer_root)


def _download_song_describer(url: str, path: str, chunk_size: int = 8192) -> None:
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


def _unpack_zip(path: str, song_describer_root: Optional[str] = None) -> None:
    root = os.path.dirname(path)

    if song_describer_root is None:
        filename = os.path.basename(path)
        filename, _ = os.path.splitext(filename)
        song_describer_root = os.path.join(root, filename)

    os.makedirs(song_describer_root, exist_ok=True)

    with zipfile.ZipFile(path) as f:
        f.extractall(song_describer_root)


if __name__ == "__main__":
    main()
