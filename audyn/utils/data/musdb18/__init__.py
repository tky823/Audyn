from ._download import download_track_names
from .dataset import MUSDB18, RandomStemsMUSDB18Dataset, StemsMUSDB18Dataset

__all__ = [
    "track_names",
    "train_track_names",
    "validation_track_names",
    "test_track_names",
    "sources",
    "accompaniments",
    "num_sources",
    "MUSDB18",
    "StemsMUSDB18Dataset",
    "RandomStemsMUSDB18Dataset",
]

track_names = download_track_names()
all_track_names = track_names
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
