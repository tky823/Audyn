import os
import shutil
import uuid
from urllib.request import Request, urlopen

from omegaconf import DictConfig

from ..utils.data.download import download_by_response
from ..utils.hydra import main as audyn_main

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


@audyn_main(config_name="download-mtg-jamendo")
def main(config: DictConfig) -> None:
    """Download MTG-Jamendo audio files.

    .. code-block:: shell

        server_type="mirror"  # or "origin"
        quality="raw"  # or "low"
        root="./MTG-Jamendo/raw"  # root directory to store
        chunk_size=8192  # chunk size in byte to download

        audyn-download-mtg-jamando \
        server_type="${server_type}" \
        quality="${quality}" \
        root="${root}" \
        chunk_size=${chunk_size}

    """
    download_mtg_jamendo(config)


def download_mtg_jamendo(config: DictConfig) -> None:
    # ported from https://github.com/MTG/mtg-jamendo-dataset/blob/1b4fa8c32e076c73b5175c1703ae805b4109309d/scripts/download/download.py  # noqa: E501
    num_files = 100

    server_type = config.server_type
    quality = config.quality
    root = config.root
    chunk_size = config.chunk_size

    if server_type == "origin":
        url = "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/raw_30s/"
    elif server_type == "mirror":
        url = "https://cdn.freesound.org/mtg-jamendo/raw_30s/"
    else:
        raise ValueError(f"{server_type} is not supported as quality. Use 'origin' or 'mirror'.")

    if quality == "raw":
        url += "audio/"
        tar_template = "raw_30s_audio-{:02d}.tar"
    elif quality == "low":
        url += "audio-low/"
        tar_template = "raw_30s_audio-low-{:02d}.tar"
    else:
        raise ValueError(f"{quality} is not supported as quality. Use 'raw' or 'low'.")

    if root is None:
        raise ValueError("Set root directory.")

    if chunk_size is None:
        chunk_size = 1024

    if root:
        os.makedirs(root, exist_ok=True)

    for idx in range(num_files):
        filename = tar_template.format(idx)
        _url = url + filename
        path = os.path.join(root, filename)
        temp_path = path + str(uuid.uuid4())[:8]

        request = Request(_url)

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


if __name__ == "__main__":
    main()
