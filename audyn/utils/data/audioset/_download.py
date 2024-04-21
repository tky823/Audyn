import json
import os
from typing import Dict, List, Optional

from ... import audyn_cache_dir
from ...github import download_file_from_github_release


def download_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, str]]:
    """Download tags of AudioSet.

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        dict: 527 tags of AudioSet.

    Examples:

        >>> from audyn.utils.data.audioset import tags
        >>> len(tags)
        527
        >>> tags[0]
        {'tag': '/m/09x0r', 'name': 'Speech'}
        >>> tags[-1]
        {'tag': '/m/07hvw1', 'name': 'Field recording'}

    """
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/audioset-tags.json"
    filename = "tags.json"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "audioset")

    path = os.path.join(root, filename)

    download_file_from_github_release(
        url,
        path,
        force_download=force_download,
        chunk_size=chunk_size,
    )

    with open(path) as f:
        tags = json.load(f)

    return tags
