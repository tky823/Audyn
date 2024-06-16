import os
from typing import List, Optional

from ....utils import audyn_cache_dir
from ...github import download_file_from_github_release


def download_speakers(
    root: Optional[str] = None,
    url: Optional[str] = None,
    version: str = "0.92",
) -> List[str]:
    """

    Args:
        version (str): Version of VCTK dataset. Only ``0.92`` is supported.

    """
    if version != "0.92":
        raise ValueError("Only version=0.92 is supported.")

    filename = f"vctk-{version}-speakers.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "vctk")

    if url is None:
        url = f"https://github.com/tky823/Audyn/releases/download/v0.0.1.dev6/vctk-{version}-speakers.txt"  # noqa: E501

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
