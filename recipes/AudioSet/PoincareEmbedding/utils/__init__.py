from ._download import download_audioset_name_to_index, download_audioset_tags
from .indexer import AudioSetIndexer

__all__ = [
    "AudioSetIndexer",
    "tags",
    "name_to_index",
    "num_tags",
]


tags = download_audioset_tags()
name_to_index = download_audioset_name_to_index()
num_tags = len(tags)
