import csv
from typing import Any, Dict, List, Tuple

import torch

from .._common import decode_csv_line as _decode_csv_line
from ._download import download_birdclef2023_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
    "decode_csv_line",
]

primary_labels = download_birdclef2023_primary_labels()
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
        indices = torch.randperm(num_files, generator=g).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames


def decode_csv_line(line: List[str]) -> Dict[str, Any]:
    """Decode line of train_metadata.csv in BirdCLEF2023.

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
    return _decode_csv_line(line, version=2023)
