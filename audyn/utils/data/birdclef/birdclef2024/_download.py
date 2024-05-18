import os
from typing import List, Optional

from .....utils import audyn_cache_dir
from ....github import download_file_from_github_release


def download_birdclef2024_primary_labels(
    root: Optional[str] = None, url: Optional[str] = None
) -> List[str]:
    filename = "primary_labels.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "birdclef2024")

    if url is None:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev4/primary_labels.txt"

    path = os.path.join(root, filename)

    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        download_file_from_github_release(url, path)

    labels = []

    with open(path) as f:
        for line in f:
            label = line.strip()
            labels.append(label)

    return labels
