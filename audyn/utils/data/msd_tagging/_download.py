import os
import zipfile
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


def download_interactions(
    type: str, root: Optional[str] = None, subset: Optional[Union[str, List[str]]] = None
) -> Dict[str, Dict[str, int]]:
    """Download interactions (user-item implicit feedbacks) of Million Song Dataset (MSD).

    Args:
        type (str): Type of interactions. Only ``user20-track200`` is supported.
            - ``user20-track200``: See details in [#liang2018variational]_.
        subset (str or list, optional): Subset(s). ``train``, ``validate-visible``,
            ``validate-hidden``, ``evaluate-visible``, and ``evaluate-hidden`` are supported.

    Returns:
        dict: Listened tracks and counts by users.

    Examples:

        >>> from audyn.utils.data.msd_tagging import download_interactions
        >>> interactions = download_interactions("user20-track200", subset="validate-visible")
        >>> len(interactions)
        50000
        >>> interactions["969cc6fb74e076a68e36a04409cb9d3765757508"]["SOABRAB12A6D4F7AAF"]
        2  # User 969cc6fb74e076a68e36a04409cb9d3765757508 listened track SOABRAB12A6D4F7AAF twice.

    .. note::

        It may take few minutes to download dataset and prepare dictionary.

    """
    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "msd")

    interactions = {}

    if type == "user20-track200":
        url = "https://github.com/tky823/Audyn/releases/download/v0.0.3/msd_user20-track200_{subset}.txt.zip"  # noqa: E501
        filename = "msd_user20-track200_{subset}.txt"

        if subset is None:
            subset = [
                "train",
                "validate-visible",
                "validate-hidden",
                "evaluate-visible",
                "evaluate-hidden",
            ]

        if isinstance(subset, str):
            subsets = [subset]
        elif isinstance(subset, list):
            subsets = subset
        else:
            raise ValueError(f"{subset} is not supported as subset.")

        for subset in subsets:
            _url = url.format(subset=subset)
            _filename = filename.format(subset=subset)
            path = os.path.join(root, _filename + ".zip")
            download_file_from_github_release(_url, path)

        for subset in subsets:
            _filename = filename.format(subset=subset)
            path = os.path.join(root, _filename + ".zip")

            with zipfile.ZipFile(path) as zf:
                with zf.open(_filename) as f:
                    for line in f:
                        line = line.decode()
                        line = line.strip()
                        user_id, track_id, count = line.split("\t")

                        count = int(count)

                        if user_id not in interactions:
                            interactions[user_id] = {}

                        interactions[user_id][track_id] = count
    else:
        raise ValueError(f"{type} is not supported as type.")

    return interactions
