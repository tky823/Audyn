import ast
import csv
import os
from typing import Any, Dict, List, Tuple

import torch

from ._download import download_birdclef2024_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
    "decode_csv_line",
]

primary_labels = download_birdclef2024_primary_labels()
num_primary_labels = len(primary_labels)


def stratified_split(
    path: str,
    train_ratio: float,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Args:
        path (str): Path to csv file.
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    g = torch.Generator()
    g.manual_seed(seed)

    filenames = {primary_label: [] for primary_label in primary_labels}
    train_filenames = []
    validation_filenames = []

    with open(path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx < 1:
                continue

            primary_label, *_, filename = line
            filenames[primary_label].append(filename)

    # split dataset
    for primary_label, _filenames in filenames.items():
        num_files = len(_filenames)
        indices = torch.randperm(num_files).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames


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
