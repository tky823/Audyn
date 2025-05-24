import glob
import os
import shutil
import tempfile
import zipfile
from typing import List, Optional, Union

from ....utils import audyn_cache_dir
from ..._github import download_file_from_github_release


def download_track_ids(
    root: Optional[str] = None,
    type: Optional[Union[int, str]] = None,
    subset: Optional[Union[str, List[str]]] = None,
) -> List[int]:
    """Download track IDs of FMA dataset.

    Args:
        root (str, optional): Root directory to save files of track IDs.
            Default: ``$HOME/.cache/audyn/data/fma``.
        type (str, optional): Dataset size. ``"small"``, ``"medium"``, ``"large"``, and
            ``"full"`` are supported.
        subset (str or list, optional): Subset name(s). ``"train"``, ``"validation"``, and
            ``"test"`` are supported.

    Returns:
        list: List of track IDs.

    """
    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "fma")

    if type is None:
        _type = "medium"
    else:
        _type = type

    assert _type in ["small", "medium", "large", "full"], f"Invalid type: {_type} is found."

    if _type == "full":
        types = ["small", "medium", "large"]
    else:
        types = [_type]

    if subset is None:
        subset = ["train", "validation", "test"]

    if isinstance(subset, str):
        subset = [subset]

    assert len(set(subset)) == len(subset), f"Duplication is found at subset={subset}."

    url = "https://github.com/tky823/Audyn/releases/download/v0.1.0/fma_track_ids.zip"
    filename = os.path.basename(url)
    path = os.path.join(root, filename)

    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        download_file_from_github_release(url, path)

    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(path, "r") as f:
            f.extractall(temp_dir)

        os.makedirs(root, exist_ok=True)

        for _temp_dir in glob.glob(os.path.join(temp_dir, "*")):
            if not os.path.isdir(_temp_dir):
                shutil.move(_temp_dir, root)

    gathered_track_ids = set()

    for _type in types:
        for _subset in subset:
            path = os.path.join(root, "track_ids", _type, f"{_subset}.txt")

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    track_id = int(line)

                    assert track_id not in gathered_track_ids

                    gathered_track_ids.add(track_id)

    gathered_track_ids = sorted(gathered_track_ids)

    return gathered_track_ids
