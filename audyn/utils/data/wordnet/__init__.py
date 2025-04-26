from ._download import load_mammal_name_to_index
from .dataloader import WordNetDataLoader
from .dataset import EvaluationMammalDataset, TrainingMammalDataset
from .indexer import WordNetIndexer

__all__ = [
    "TrainingMammalDataset",
    "EvaluationMammalDataset",
    "WordNetDataLoader",
    "WordNetIndexer",
    "load_mammal_name_to_index",
]
