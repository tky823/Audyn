import json
import os
from typing import List, Optional, Union

from ....utils import audyn_cache_dir
from ..._github import download_file_from_github_release


def download_track_names(
    root: Optional[str] = None,
    version: Optional[Union[int, str]] = None,
    subset: Optional[Union[str, List[str]]] = None,
    url: Optional[str] = None,
) -> List[int]:
    """Download all track names of DnR dataset.

    Args:
        root (str, optional): Root directory to save ``dnr-v{version}_track_names.json``.
            Default: ``$HOME/.cache/audyn/data/dnr-v{version}``.
        version (int or str, optional)
        subset (str or list, optional): Subset name(s). ``train``, ``validation``, and ``test`` are supported.
        url (str, optional): URL of pre-defined ``dnr-v{version}_track_names.json``.
            Default: ``https://github.com/tky823/Audyn/releases/download/v0.0.5/dnr-v{version}_track_names.json``.  # noqa: E501

    Returns:
        list: List of track names.

    """
    if version is None:
        version = "2"
    else:
        version = str(version)

    assert version == "2", "Only v2 is supported."

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", f"dnr-v{version}")

    if subset is None:
        subset = ["train", "validation", "test"]

    if isinstance(subset, str):
        subset = [subset]

    assert len(set(subset)) == len(subset), "Duplication is found at subset={}.".format(subset)

    if url is None:
        url = f"https://github.com/tky823/Audyn/releases/download/v0.0.5/dnr-v{version}_track_names.json"  # noqa: E501

    filename = os.path.basename(url)
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        download_file_from_github_release(url, path)

    with open(path) as f:
        track_names = json.load(f)

    gathered_track_names = []

    for _subset in subset:
        gathered_track_names += track_names[_subset]

    return gathered_track_names
