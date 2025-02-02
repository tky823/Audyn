from ._download import download_all_metadata, download_all_tags

__all__ = [
    "all_tags",
    "num_all_tags",
    "download_all_metadata",
]

all_tags = download_all_tags()
num_all_tags = len(all_tags)
