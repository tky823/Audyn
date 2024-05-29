from .birdclef2022 import num_primary_labels as num_birdclef2022_primary_labels
from .birdclef2022 import primary_labels as birdclef2022_primary_labels
from .birdclef2023 import num_primary_labels as num_birdclef2023_primary_labels
from .birdclef2023 import primary_labels as birdclef2023_primary_labels
from .birdclef2023.dataset import BirdCLEF2023PrimaryLabelDataset
from .birdclef2024 import num_primary_labels as num_birdclef2024_primary_labels
from .birdclef2024 import primary_labels as birdclef2024_primary_labels
from .birdclef2024.collator import BirdCLEF2024BaselineCollator
from .birdclef2024.composer import BirdCLEF2024AudioComposer, BirdCLEF2024PrimaryLabelComposer
from .birdclef2024.dataset import BirdCLEF2024AudioDataset, BirdCLEF2024PrimaryLabelDataset

__all__ = [
    # for BirdCLEF2022
    "birdclef2022_primary_labels",
    "num_birdclef2022_primary_labels",
    # for BirdCLEF2023
    "birdclef2023_primary_labels",
    "num_birdclef2023_primary_labels",
    "BirdCLEF2023PrimaryLabelDataset",
    # for BirdCLEF2024
    "birdclef2024_primary_labels",
    "num_birdclef2024_primary_labels",
    "BirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024AudioDataset",
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024AudioComposer",
    "BirdCLEF2024BaselineCollator",
]
