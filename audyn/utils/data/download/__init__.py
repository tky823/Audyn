import os
from io import BufferedWriter
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False


def download_file(
    url: str,
    path: str,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> None:
    """Download file from url.

    Args:
        url (str): URL to download.
        path (str): Path to save file.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    """
    if os.path.exists(path):
        if force_download:
            os.remove(path)
        else:
            return

    download_dir = os.path.dirname(path)

    if download_dir:
        os.makedirs(download_dir, exist_ok=True)

    request = Request(url)

    try:
        with urlopen(request) as response, open(path, "wb") as f:
            total_size = int(response.headers["Content-Length"])

            if IS_TQDM_AVAILABLE:
                description = f"Download file to {path}"

                with tqdm(unit="B", unit_scale=True, desc=description, total=total_size) as pbar:
                    download_by_response(response, f, chunk_size=chunk_size, pbar=pbar)
            else:
                download_by_response(response, f, chunk_size=chunk_size)
    except Exception as e:
        raise e


def download_by_response(response, f: BufferedWriter, chunk_size: int = 1024, pbar=None) -> None:
    while True:
        chunk = response.read(chunk_size)

        if not chunk:
            break

        f.write(chunk)

        if pbar is not None:
            pbar.update(len(chunk))
