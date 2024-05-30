from .._common import decode_csv_line
from ._download import download_birdclef2021_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
    "decode_csv_line",  # for compatibility with birdclef2024
]

primary_labels = download_birdclef2021_primary_labels()
num_primary_labels = len(primary_labels)
