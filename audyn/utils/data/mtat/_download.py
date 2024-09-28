from typing import List


def download_tags() -> List[str]:
    """Download MTAT tags.

    Returns:
        list: List of tags.

    Examples:

        >>> from audyn.utils.data.mtat import tags
        >>> len(tags)
        50
        >>> tags[0]
        'guitar'
        >>> tags[-1]
        'choral'

    """
    tags = [
        "guitar",
        "classical",
        "slow",
        "techno",
        "strings",
        "drums",
        "electronic",
        "rock",
        "fast",
        "piano",
        "ambient",
        "beat",
        "violin",
        "vocal",
        "synth",
        "female",
        "indian",
        "opera",
        "male",
        "singing",
        "vocals",
        "no vocals",
        "harpsichord",
        "loud",
        "quiet",
        "flute",
        "woman",
        "male vocal",
        "no vocal",
        "pop",
        "soft",
        "sitar",
        "solo",
        "man",
        "classic",
        "choir",
        "voice",
        "new age",
        "dance",
        "male voice",
        "female vocal",
        "beats",
        "harp",
        "cello",
        "no voice",
        "weird",
        "country",
        "metal",
        "female voice",
        "choral",
    ]

    return tags
