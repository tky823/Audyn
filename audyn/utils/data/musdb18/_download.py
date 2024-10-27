import os
from typing import List, Optional, Union

from ....utils import audyn_cache_dir
from ..._github import download_file_from_github_release


def download_track_names(
    root: Optional[str] = None,
    subset: Optional[Union[str, List[str]]] = None,
    url: Optional[str] = None,
) -> List[str]:
    """Download all track names of MUSDB18 dataset.

    Args:
        root (str, optional): Rootdirectory to save ``musdb18_track_names.txt``.
            Default: ``$HOME/.cache/audyn/data/musdb18``.
        subset (str or list, optional): Subset name(s). ``train``, ``validation``, and ``test`` are supported.
        url (str, optional): URL of pre-defined ``musdb18_track_names.txt``.
            Default: ``https://github.com/tky823/Audyn/releases/download/v0.0.1.dev8/musdb18_track_names.txt``.  # noqa: E501

    Returns:
        list: List of track names.

    """
    filename = "musdb18_track_names.txt"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "musdb18")

    if subset is None:
        subset = ["train", "validation", "test"]

    if isinstance(subset, str):
        subset = [subset]

    assert len(set(subset)) == len(subset), "Duplication is found at subset={}.".format(subset)

    if url is None:
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev8/musdb18_track_names.txt"  # noqa: E501

    track_names = _download_track_names(root, url, filename)
    gathered_track_names = []

    for _subset in subset:
        if _subset == "train":
            start = 0
            end = 86
        elif _subset == "validation":
            start = 86
            end = 100
        elif _subset == "test":
            start = 100
            end = 150
        else:
            raise ValueError(f"{_subset} is not supported as subset.")

        gathered_track_names += track_names[start:end]

    return gathered_track_names


def _download_track_names(root: str, url: str, filename: str) -> List[str]:
    """Download track names of MUSDB18 dataset."""
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        download_file_from_github_release(url, path)

    track_names = []

    with open(path) as f:
        for line in f:
            label = line.strip()
            track_names.append(label)

    return track_names
