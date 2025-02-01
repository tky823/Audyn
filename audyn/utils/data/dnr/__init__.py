from ._download import download_track_names
from .dataset import DNR, RandomStemsDNRDataset, StemsDNRDataset, Track

__all__ = [
    "v2_all_track_names",
    "v2_train_track_names",
    "v2_validation_track_names",
    "v2_test_track_names",
    "sources",
    "num_sources",
    "DNR",
    "StemsDNRDataset",
    "RandomStemsDNRDataset",
    "Track",
]

_dnr_version = 2
v2_all_track_names = download_track_names(version=_dnr_version)
v2_train_track_names = download_track_names(version=_dnr_version, subset="train")
v2_validation_track_names = download_track_names(version=_dnr_version, subset="validation")
v2_test_track_names = download_track_names(version=_dnr_version, subset="test")
sources = ["speech", "music", "effect"]
num_sources = len(sources)
