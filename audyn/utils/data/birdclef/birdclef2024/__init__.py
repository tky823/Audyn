import csv
from typing import List, Tuple

import torch

from ._download import download_birdclef2024_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "stratified_split",
]

primary_labels = download_birdclef2024_primary_labels()
num_primary_labels = len(primary_labels)


def stratified_split(
    path: str,
    train_ratio: float,
    seed: int = 0,
    shuffle_train: bool = True,
    shuffle_validation: bool = False,
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

    num_train_files = len(train_filenames)
    num_validation_files = len(validation_filenames)
    train_indices = torch.randperm(num_train_files, generator=g).tolist()
    validation_indices = torch.randperm(num_validation_files, generator=g).tolist()

    if shuffle_train:
        train_filenames = [train_filenames[idx] for idx in train_indices]

    if shuffle_validation:
        validation_filenames = [validation_filenames[idx] for idx in validation_indices]

    return train_filenames, validation_filenames
