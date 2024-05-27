import ast
import os
from typing import Any, Dict, List

__all__ = [
    "decode_csv_line",
]


def decode_csv_line(line: List[str]) -> Dict[str, Any]:
    """Decode line of train_metadata.csv.

    Args:
        line (list): One line of train_metadata.csv split by comma (,).

    Returns:
        dict: Dictionary containing metadata of given line.

    .. note::

        Returned dictionary contains following values.

            - filename (str): Filename with out extension. e.g. ``asbfly/XC134896``.
            - primary_label (str): Primary label of bird species.
            - secondary_label (list): Secondary labels of bird species.
            - type (list): Chirp types.
            - latitude (float, optional): Latitude of recording.
            - longitude (float, optional): Longitude of recording.
            - scientific_name (str): Scientific name of bird.
            - common_name (str): Common name of bird.
            - rating (float): Rating.
            - path (str): Path to audio file equivalent to ``filename`` + ``.ogg``.
                e.g. ``asbfly/XC134896.ogg``.

    """
    (
        primary_label,
        secondary_labels,
        chirp_types,
        latitude,
        longitude,
        scientific_name,
        common_name,
        _,
        _,
        rating,
        _,
        path,
    ) = line

    secondary_labels = ast.literal_eval(secondary_labels)
    chirp_types = ast.literal_eval(chirp_types)
    secondary_labels = [secondary_label.lower() for secondary_label in secondary_labels]
    chirp_types = [chirp_type.lower() for chirp_type in chirp_types]

    filename, _ = os.path.splitext(path)

    if len(latitude) > 0:
        latitude = float(latitude)
    else:
        latitude = None

    if len(longitude) > 0:
        longitude = float(longitude)
    else:
        longitude = None

    data = {
        "filename": filename,
        "primary_label": primary_label,
        "secondary_label": secondary_labels,
        "type": chirp_types,
        "latitude": latitude,
        "longitude": longitude,
        "scientific_name": scientific_name,
        "common_name": common_name,
        "rating": float(rating),
        "path": path,
    }

    return data
