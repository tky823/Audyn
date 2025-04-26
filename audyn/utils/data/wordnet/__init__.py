from ._download import load_mammal_name_to_index
from .dataloader import WordNetDataLoader
from .dataset import EvaluationMammalDataset, TrainingMammalDataset

__all__ = [
    "TrainingMammalDataset",
    "EvaluationMammalDataset",
    "WordNetDataLoader",
    "load_mammal_name_to_index",
]
