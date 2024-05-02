import json
import os
from io import BufferedWriter
from typing import Tuple
from urllib.parse import urlparse
from urllib.request import Request, urlopen

try:
    from tqdm import tqdm

    IS_TQDM_AVAILABLE = True
except ImportError:
    IS_TQDM_AVAILABLE = False

__all__ = ["download_file_from_github_release"]


def download_file_from_github_release(
    url: str,
    path: str,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> None:
    """Download file from github release.

    Args:
        url (str): URL of assets displayed in browser.
        path (str): Path to save file.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    .. note::

        You may need to set ``GITHUB_TOKEN`` as environmental variable to download
        private assets.

    """
    token = os.getenv("GITHUB_TOKEN", None)

    if os.path.exists(path):
        if force_download:
            os.remove(path)
        else:
            return

    download_dir = os.path.dirname(path)

    if download_dir:
        os.makedirs(download_dir, exist_ok=True)

    headers = {
        "Accept": "application/octet-stream",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    url, total_size = _obtain_metadata(url)
    request = Request(url, headers=headers)

    try:
        with urlopen(request) as response, open(path, "wb") as f:
            if IS_TQDM_AVAILABLE:
                description = f"Download file to {path}"

                with tqdm(unit="B", unit_scale=True, desc=description, total=total_size) as pbar:
                    _download(response, f, chunk_size=chunk_size, pbar=pbar)
            else:
                _download(response, f, chunk_size=chunk_size)
    except Exception as e:
        raise e


def _obtain_metadata(url: str) -> Tuple[str, int]:
    """Convert browser_download_url to actual url to download."""
    token = os.getenv("GITHUB_TOKEN", None)

    parsed_url = urlparse(url)
    parsed_url.hostname
    _, owner, repo, _, _, tag, _ = parsed_url.path.split("/")

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if token is not None:
        headers["Authorization"] = f"Bearer {token}"

    endpoint = f"https://api.github.com/repos/{owner}/{repo}/releases"
    request = Request(endpoint, headers=headers)

    with urlopen(request) as response:
        data = response.read().decode("utf-8")

    data = json.loads(data)

    converted_url = None
    total_size = None
    is_found = False

    for release in data:
        if tag == release["tag_name"]:
            for asset in release["assets"]:
                if url == asset["browser_download_url"]:
                    converted_url = asset["url"]
                    total_size = asset["size"]
                    is_found = True

                if is_found:
                    break

        if is_found:
            break

    if not is_found:
        raise ValueError(f"Asset {url} is not found.")

    return converted_url, total_size


def _download(response, f: BufferedWriter, chunk_size: int = 1024, pbar=None) -> None:
    while True:
        chunk = response.read(chunk_size)

        if not chunk:
            break

        f.write(chunk)

        if pbar is not None:
            pbar.update(len(chunk))
