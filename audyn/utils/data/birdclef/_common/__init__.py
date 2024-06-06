import ast
import os
from typing import Any, Dict, List, Optional, Union

__all__ = [
    "decode_csv_line",
]


def decode_csv_line(
    line: List[str],
    version: Optional[Union[str, int]] = None,
) -> Dict[str, Any]:
    """Decode line of train_metadata.csv.

    Args:
        line (list): One line of train_metadata.csv split by comma (,).
        version (str or int, optional): Version information.

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
    if version is None:
        if len(line) == 12:
            version = 2023
        elif len(line) == 13:
            version = 2022
        elif len(line) == 14:
            version = 2021
        else:
            raise ValueError("Invalid format of line is detected.")

    version = int(version)

    if version == 2021:
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
            filename,
            _,
            rating,
            _,
            _,
        ) = line
        path = os.path.join(primary_label, filename)
    elif version == 2022:
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
            _,
            path,
        ) = line
    elif version in [2023, 2024]:
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
    else:
        raise ValueError("Invalid format of line is detected.")

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
