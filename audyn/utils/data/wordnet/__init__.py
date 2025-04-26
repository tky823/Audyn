from ._download import (
    download_mammal_name_to_index,
    download_mammal_tags,
    download_mammal_taxonomy,
)
from .composer import WordNetComposer
from .dataloader import WordNetDataLoader
from .dataset import EvaluationMammalDataset, TrainingMammalDataset
from .indexer import WordNetIndexer

__all__ = [
    "TrainingMammalDataset",
    "EvaluationMammalDataset",
    "WordNetDataLoader",
    "WordNetIndexer",
    "WordNetComposer",
    "mammal_tags",
    "mammal_taxonomy",
    "mammal_name_to_index",
    "num_mammal_tags",
]

mammal_tags = download_mammal_tags()
mammal_taxonomy = download_mammal_taxonomy()
mammal_name_to_index = download_mammal_name_to_index()
num_mammal_tags = len(mammal_tags)
