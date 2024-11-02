from ._download import download_metadata, download_tags

__all__ = [
    "tags",
    "num_tags",
    "download_tags",
    "download_metadata",
]

tags = download_tags()
num_tags = len(tags)
