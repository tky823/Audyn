import os
from typing import List, Optional

from ... import audyn_cache_dir
from ...github import download_file_from_github_release


def download_top50_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[str]:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1/mtg-jamendo_top50-tags.txt"  # noqa: E501
    filename = "top50-tags.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo")

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


def download_genre_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[str]:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1/mtg-jamendo_genre-tags.txt"  # noqa: E501
    filename = "genre-tags.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo")

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


def download_instrument_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[str]:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1/mtg-jamendo_instrument-tags.txt"  # noqa: E501
    filename = "instrument-tags.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo")

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


def download_moodtheme_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[str]:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.1/mtg-jamendo_moodtheme-tags.txt"  # noqa: E501
    filename = "moodtheme-tags.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "mtg-jamendo")

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
