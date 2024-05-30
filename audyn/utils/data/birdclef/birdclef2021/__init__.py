from typing import Any, Dict, List

from .._common import decode_csv_line as _decode_csv_line
from ._download import download_birdclef2021_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "decode_csv_line",
]

primary_labels = download_birdclef2021_primary_labels()
num_primary_labels = len(primary_labels)


def decode_csv_line(line: List[str]) -> Dict[str, Any]:
    """Decode line of train_metadata.csv in BirdCLEF2021.

    Args:
        line (list): One line of train_metadata.csv split by comma (,).

    Returns:
        dict: Dictionary containing metadata of given line.

    .. note::

        Returned dictionary contains following values.

            - filename (str): Filename with out extension. e.g. ``abethr1/XC128013``.
            - primary_label (str): Primary label of bird species.
            - secondary_label (list): Secondary labels of bird species.
            - type (list): Chirp types.
            - latitude (float, optional): Latitude of recording.
            - longitude (float, optional): Longitude of recording.
            - scientific_name (str): Scientific name of bird.
            - common_name (str): Common name of bird.
            - rating (float): Rating.
            - path (str): Path to audio file equivalent to ``filename`` + ``.ogg``.
                e.g. ``abethr1/XC128013.ogg``.

    """
    return _decode_csv_line(line, version=2021)
