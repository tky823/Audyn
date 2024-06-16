from ._download import download_speakers

__all__ = [
    "speakers",
    "num_speakers",
]

speakers = download_speakers()
num_speakers = len(speakers)
