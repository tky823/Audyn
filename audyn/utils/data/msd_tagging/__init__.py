from ._download import download_tags

__all__ = [
    "tags",
    "num_tags",
    "download_tags",
]

tags = download_tags()
num_tags = len(tags)
