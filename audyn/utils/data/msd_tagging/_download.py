import json
import os
from typing import Dict, List, Optional, Union

from ... import audyn_cache_dir
from ..._github import download_file_from_github_release


def download_tags() -> List[Dict[str, str]]:
    """Download tags of Million Song Dataset (MSD).

    Returns:
        list: 50 tags of MSD.

    Examples:

        >>> from audyn.utils.data.msd_tagging import tags
        >>> len(tags)
        50
        >>> tags[0]
        'rock'
        >>> tags[-1]
        'progressive metal'

    """
    tags = [
        "rock",
        "pop",
        "indie",
        "alternative",
        "electronic",
        "hip-hop",
        "metal",
        "jazz",
        "punk",
        "folk",
        "alternative rock",
        "indie rock",
        "dance",
        "hard rock",
        "00s",
        "soul",
        "hardcore",
        "80s",
        "country",
        "classic rock",
        "punk rock",
        "blues",
        "chillout",
        "experimental",
        "heavy metal",
        "death metal",
        "90s",
        "reggae",
        "progressive rock",
        "ambient",
        "acoustic",
        "beautiful",
        "british",
        "rnb",
        "funk",
        "metalcore",
        "mellow",
        "world",
        "guitar",
        "trance",
        "indie pop",
        "christian",
        "house",
        "spanish",
        "latin",
        "psychedelic",
        "electro",
        "piano",
        "70s",
        "progressive metal",
    ]

    return tags


def download_metadata(
    root: Optional[str] = None,
    subset: Optional[Union[str, List[str]]] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, str]]:
    """Download metadata of MSD dataset.

    Args:
        subset (str or list, optional): Subset name(s). ``labeled-train``, ``unlabeled-train``,
            ``validate``, ``evaluate`` are supported.

    Returns:
        list: List of metadata.

    Examples:

        >>> from audyn.utils.data.msd_tagging import download_metadata
        >>> metadata = download_metadata(subset="labeled-train")
        >>> len(metadata)
        163504

    """
    base_url = "https://github.com/tky823/Audyn/releases/download/v0.0.3/msd_{subset}.jsonl"

    if subset is None:
        subsets = ["train", "validation", "test"]
    elif isinstance(subset, str):
        subsets = [subset]
    else:
        subsets = subset

    metadata = []

    for subset in subsets:
        url = base_url.format(subset=subset)
        filename = "msd_{subset}.jsonl".format(subset=subset)

        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "msd-tagging")

        path = os.path.join(root, filename)

        download_file_from_github_release(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )
        _metadata = _load_metadata(path)
        metadata.extend(_metadata)

    return metadata


def _load_metadata(path: str) -> List[Dict[str, str]]:
    metadata = []

    with open(path) as f:
        for line in f:
            _metadata = json.loads(line)
            metadata.append(_metadata)

    return metadata
