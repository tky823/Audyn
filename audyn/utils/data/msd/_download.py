import os
from typing import Dict, List, Optional

from ... import audyn_cache_dir
from ...github import download_file_from_github_release


def download_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, str]]:
    """Download tags of Million Song Dataset (MSD).

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        dict: 50 tags of MSD>

    Examples:

        >>> from audyn.utils.data.msd import tags
        >>> len(tags)
        50
        >>> tags[0]
        'rock'
        >>> tags[-1]
        'progressive metal'

    """
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.2/msd-tags.txt"
    filename = "msd-tags.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "msd")

    path = os.path.join(root, filename)

    download_file_from_github_release(
        url,
        path,
        force_download=force_download,
        chunk_size=chunk_size,
    )

    tags = []

    with open(path) as f:
        for line in f:
            tag = line.strip()
            tags.append(tag)

    return tags
