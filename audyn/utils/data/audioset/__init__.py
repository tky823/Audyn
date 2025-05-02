from ._download import download_name_to_index, download_tag_to_index, download_tags
from .indexer import AudioSetIndexer

__all__ = [
    "AudioSetIndexer",
    "tags",
    "tag_to_index",
    "name_to_index",
    "num_tags",
]

tags = download_tags()
tag_to_index = download_tag_to_index()
name_to_index = download_name_to_index()
num_tags = len(tags)
