from ._download import download_interactions, download_tags

__all__ = [
    "tags",
    "num_tags",
    "download_interactions",
]

tags = download_tags()
num_tags = len(tags)
