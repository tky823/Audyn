from typing import Dict, List


def download_tags() -> List[Dict[str, str]]:
    """Download tags of Million Song Dataset (MSD).

    Args:
        root (str, optional): Root directory to download tags.
        force_download (str): If ``True``, existing file is overwritten by new one.
        chunk_size (int): Chunk size to download file.

    Returns:
        list: 50 tags of MSD.

    Examples:

        >>> from audyn.utils.data.msd import tags
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
