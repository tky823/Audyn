from ._download import download_track_names

__all__ = [
    "track_names",
    "train_track_names",
    "validation_track_names",
    "test_track_names",
]

track_names = download_track_names()
all_track_names = track_names
train_track_names = download_track_names(subset="train")
validation_track_names = download_track_names(subset="validation")
test_track_names = download_track_names(subset="test")
