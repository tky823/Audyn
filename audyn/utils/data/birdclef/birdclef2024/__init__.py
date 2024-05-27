import csv
import glob
import os
from typing import List, Tuple

import torch

from .._common import decode_csv_line
from ._download import download_birdclef2024_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
    "split",
    "decode_csv_line",  # for backward compatibility
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
