from ._download import download_track_ids
from .dataset import FreeMusicArchiveNAFPDataset

__all__ = [
    "small_train_track_ids",
    "small_validation_track_ids",
    "small_test_track_ids",
    "medium_train_track_ids",
    "medium_validation_track_ids",
    "medium_test_track_ids",
    "large_train_track_ids",
    "large_validation_track_ids",
    "large_test_track_ids",
    "full_train_track_ids",
    "full_validation_track_ids",
    "full_test_track_ids",
    "FreeMusicArchiveNAFPDataset",
]


small_train_track_ids = download_track_ids(type="small", subset="train")
small_validation_track_ids = download_track_ids(type="small", subset="validation")
small_test_track_ids = download_track_ids(type="small", subset="test")
medium_train_track_ids = download_track_ids(type="medium", subset="train")
medium_validation_track_ids = download_track_ids(type="medium", subset="validation")
medium_test_track_ids = download_track_ids(type="medium", subset="test")
large_train_track_ids = download_track_ids(type="large", subset="train")
large_validation_track_ids = download_track_ids(type="large", subset="validation")
large_test_track_ids = download_track_ids(type="large", subset="test")
full_train_track_ids = download_track_ids(type="full", subset="train")
full_validation_track_ids = download_track_ids(type="full", subset="validation")
full_test_track_ids = download_track_ids(type="full", subset="test")
