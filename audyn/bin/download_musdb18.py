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


@audyn_main(config_name="download-musdb18")
def main(config: DictConfig) -> None:
    r"""Download MUSDB18 dataset.

    .. code-block:: shell

        type="default"  # for MUSDB18
        # type="hq"  # for MUSDB18-HQ
        # type="7s"  # for MUSDB18-7s

        musdb18_root="./MUSDB18"  # root directory to save dataset
        chunk_size=1024  # chunk size in byte to download

        audyn-download-musdb18 \
        type="${type}" \
        root="${musdb18_root}" \
        chunk_size=${chunk_size}

    """
    download_musdb18(config)


def download_musdb18(config: DictConfig) -> None:
    _type = config.type
    root = config.root
    chunk_size = config.chunk_size

    if _type is None:
        _type = "default"

    if chunk_size is None:
        chunk_size = 1024

    if _type == "default":
        url = "https://zenodo.org/records/1117372/files/musdb18.zip"
    elif _type == "hq":
        url = "https://zenodo.org/records/3338373/files/musdb18hq.zip"
    elif _type == "7s":
        url = "https://zenodo.org/api/files/1ff52183-071a-4a59-923f-7a31c4762d43/MUSDB18-7-STEMS.zip"  # noqa: E501
    else:
        raise RuntimeError(f"{_type} is not supported as type. Choose default, hq, or 7s.")

    filename = os.path.basename(url)
    path = os.path.join(root, filename)
    temp_path = path + str(uuid.uuid4())[:8]

    if root:
        os.makedirs(root, exist_ok=True)

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


if __name__ == "__main__":
    main()
