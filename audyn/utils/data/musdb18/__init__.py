from ._download import download_track_names
from .dataset import (
    MUSDB18,
    DistributedRandomStemsMUSDB18Dataset,
    RandomStemsMUSDB18Dataset,
    StemsMUSDB18Dataset,
    Track,
)

__all__ = [
    "all_track_names",
    "train_track_names",
    "validation_track_names",
    "test_track_names",
    "sources",
    "accompaniments",
    "num_sources",
    "MUSDB18",
    "StemsMUSDB18Dataset",
    "RandomStemsMUSDB18Dataset",
    "DistributedRandomStemsMUSDB18Dataset",
    "Track",
]

all_track_names = download_track_names()
train_track_names = download_track_names(subset="train")
validation_track_names = download_track_names(subset="validation")
test_track_names = download_track_names(subset="test")
accompaniments = [
    "drums",
    "bass",
    "other",
]
sources = accompaniments + ["vocals"]
num_sources = len(sources)
