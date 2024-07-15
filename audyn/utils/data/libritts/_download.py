import os
from typing import List, Optional

from ....utils import audyn_cache_dir
from ...github import download_file_from_github_release


def download_speakers(root: Optional[str] = None, url: Optional[str] = None) -> List[str]:
    """Download all speakers of LibriTTS dataset."""
    filename = "libritts_speakers.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "libritts")

    if url is None:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev8/libritts_speakers.txt"  # noqa: E501

    speakers = _download_speakers(root, url, filename)

    return speakers


def _download_speakers(root: str, url: str, filename: str) -> List[str]:
    """Download speakers of LibriTTS dataset."""
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        download_file_from_github_release(url, path)

    speakers = []

    with open(path) as f:
        for line in f:
            label = line.strip()
            speakers.append(label)

    return speakers
