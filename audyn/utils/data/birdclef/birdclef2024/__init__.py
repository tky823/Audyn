import csv
import glob
import os
from typing import Any, Dict, List, Tuple

import torch

from .._common import decode_csv_line as _decode_csv_line
from ._download import download_birdclef2024_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
    "split",
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
        indices = torch.randperm(num_files, generator=g).tolist()

        for idx in indices[: int(train_ratio * num_files)]:
            train_filenames.append(_filenames[idx])

        for idx in indices[int(train_ratio * num_files) :]:
            validation_filenames.append(_filenames[idx])

    return train_filenames, validation_filenames


def split(
    path: str,
    train_ratio: float,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Split dataset into training and validation.

    Unlike ``stratified_split``, ``split`` is available for unlabeled data.

    Args:
        path (str): Path to csv file (for labeled data) or audio directory (for unlabeled data).
        train_ratio (float): Ratio of training set.
        seed (int): Random seed.

    Returns:
        tuple: Splits of filenames.

            - list: List of training filenames.
            - list: List of validation filenames.

    """
    g = torch.Generator()
    g.manual_seed(seed)

    train_filenames = []
    validation_filenames = []
    filenames = []

    if path.endswith(".csv") and os.path.isfile(path):
        with open(path) as f:
            reader = csv.reader(f)

            for idx, line in enumerate(reader):
                if idx < 1:
                    continue

                *_, filename = line
                filenames.append(filename)
    else:
        root = path
        filenames = sorted(glob.glob(os.path.join(root, "*.ogg")))
        filenames = [os.path.relpath(filename, root) for filename in filenames]

    num_files = len(filenames)
    indices = torch.randperm(num_files, generator=g)
    indices = indices.tolist()
    train_filenames = [filenames[idx] for idx in indices[: int(num_files * train_ratio)]]
    validation_filenames = [filenames[idx] for idx in indices[int(num_files * train_ratio) :]]

    return train_filenames, validation_filenames


def decode_csv_line(line: List[str]) -> Dict[str, Any]:
    """Decode line of train_metadata.csv in BirdCLEF2024.

    Args:
        line (list): One line of train_metadata.csv split by comma (,).

    Returns:
        dict: Dictionary containing metadata of given line.

    .. note::

        Returned dictionary contains following values.

            - filename (str): Filename with out extension. e.g. ``asbfly/XC49755``.
            - primary_label (str): Primary label of bird species.
            - secondary_label (list): Secondary labels of bird species.
            - type (list): Chirp types.
            - latitude (float, optional): Latitude of recording.
            - longitude (float, optional): Longitude of recording.
            - scientific_name (str): Scientific name of bird.
            - common_name (str): Common name of bird.
            - rating (float): Rating.
            - path (str): Path to audio file equivalent to ``filename`` + ``.ogg``.
                e.g. ``asbfly/XC49755.ogg``.

    """
    return _decode_csv_line(line, version=2024)
