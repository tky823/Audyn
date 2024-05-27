from ._download import download_birdclef2022_primary_labels

__all__ = [
    "primary_labels",
    "num_primary_labels",
]

primary_labels = download_birdclef2022_primary_labels()
num_primary_labels = len(primary_labels)
