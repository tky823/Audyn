import json
import os
from typing import Dict, List, Optional

from ... import audyn_cache_dir
from ..._download import DEFAULT_CHUNK_SIZE
from ..._github import download_file_from_github_release


def download_tag_to_index(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, int]:
    """Download mapping of tag to index of AudioSet.

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        dict: Mapping of tag to index of AudioSet.

    Examples:

        >>> from audyn.utils.data.audioset import tag_to_index
        >>> len(tag_to_index)
        527
        >>> tag_to_index["/m/09x0r"]
        0
        >>> tag_to_index["/m/07hvw1"]
        526

    """
    tags = download_tags(root, force_download=force_download, chunk_size=chunk_size)
    tag_to_index = {}

    for index, tag in enumerate(tags):
        _tag = tag["tag"]

        assert _tag not in tag_to_index

        tag_to_index[_tag] = index

    return tag_to_index


def download_name_to_index(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Dict[str, int]:
    """Download mapping of name to index of AudioSet.

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        dict: Mapping of tag name to index of AudioSet.

    Examples:

        >>> from audyn.utils.data.audioset import name_to_index
        >>> len(name_to_index)
        527
        >>> name_to_index["Speech"]
        0
        >>> name_to_index["Field recording"]
        526

    """
    tags = download_tags(root, force_download=force_download, chunk_size=chunk_size)
    name_to_index = {}

    for index, tag in enumerate(tags):
        _tag = tag["name"]

        assert _tag not in name_to_index

        name_to_index[_tag] = index

    return name_to_index


def download_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[Dict[str, str]]:
    """Download tags of AudioSet.

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        list: 527 tags of AudioSet.

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


def download_names(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[Dict[str, str]]:
    """Download names of AudioSet.

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        list: 527 names of AudioSet.

    Examples:

        >>> from audyn.utils.data.audioset import names
        >>> len(names)
        527
        >>> names[0]
        'Speech'
        >>> names[-1]
        'Field recording'

    """
    tags = download_tags(root=root, force_download=force_download, chunk_size=chunk_size)
    names = []

    for tag in tags:
        _tag = tag["name"]
        names.append(_tag)

    return names
