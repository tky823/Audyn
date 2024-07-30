import os
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
        output="./MTG-Jamendo/raw"  # output directory to store
        chunk_size=1024  # chunk size in byte to download

        audyn-download-mtg-jamando \
        server_type="${server_type}" \
        quality="${quality}" \
        output="${output}" \
        chunk_size=${chunk_size}

    """
    # ported from https://github.com/MTG/mtg-jamendo-dataset/blob/1b4fa8c32e076c73b5175c1703ae805b4109309d/scripts/download/download.py  # noqa: E501
    num_files = 100

    server_type = config.server_type
    quality = config.quality
    output = config.output
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

    if output is None:
        raise ValueError("Set output directory.")

    if chunk_size is None:
        chunk_size = 1024

    if output:
        os.makedirs(output, exist_ok=True)

    for idx in range(num_files):
        filename = tar_template.format(idx)
        _url = url + filename
        path = os.path.join(output, filename)

        request = Request(_url)

        try:
            with urlopen(request) as response, open(path, "wb") as f:
                if IS_TQDM_AVAILABLE:
                    total_size = int(response.headers["Content-Length"])

                    with tqdm(unit="B", unit_scale=True, desc=filename, total=total_size) as pbar:
                        download_by_response(response, f, chunk_size=chunk_size, pbar=pbar)
                else:
                    download_by_response(response, f, chunk_size=chunk_size)
        except Exception as e:
            raise e


if __name__ == "__main__":
    main()
