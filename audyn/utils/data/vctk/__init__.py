from ._download import download_speakers, download_valid_speakers

__all__ = [
    "speakers",
    "valid_speakers",
    "num_speakers",
    "num_valid_speakers",
]

speakers = download_speakers()
valid_speakers = download_valid_speakers()
num_speakers = len(speakers)
num_valid_speakers = len(valid_speakers)
