from .birdclef2021 import num_primary_labels as num_birdclef2021_primary_labels
from .birdclef2021 import primary_labels as birdclef2021_primary_labels
from .birdclef2021.collator import BirdCLEF2021BaselineCollator
from .birdclef2021.composer import BirdCLEF2021PrimaryLabelComposer
from .birdclef2021.dataset import BirdCLEF2021PrimaryLabelDataset
from .birdclef2022 import num_primary_labels as num_birdclef2022_primary_labels
from .birdclef2022 import primary_labels as birdclef2022_primary_labels
from .birdclef2022.collator import BirdCLEF2022BaselineCollator
from .birdclef2022.composer import BirdCLEF2022PrimaryLabelComposer
from .birdclef2022.dataset import BirdCLEF2022PrimaryLabelDataset
from .birdclef2023 import num_primary_labels as num_birdclef2023_primary_labels
from .birdclef2023 import primary_labels as birdclef2023_primary_labels
from .birdclef2023.collator import BirdCLEF2023BaselineCollator
from .birdclef2023.composer import BirdCLEF2023PrimaryLabelComposer
from .birdclef2023.dataset import BirdCLEF2023PrimaryLabelDataset
from .birdclef2024 import num_primary_labels as num_birdclef2024_primary_labels
from .birdclef2024 import primary_labels as birdclef2024_primary_labels
from .birdclef2024.collator import BirdCLEF2024BaselineCollator
from .birdclef2024.composer import BirdCLEF2024AudioComposer, BirdCLEF2024PrimaryLabelComposer
from .birdclef2024.dataset import BirdCLEF2024AudioDataset, BirdCLEF2024PrimaryLabelDataset

__all__ = [
    # for BirdCLEF2021
    "birdclef2021_primary_labels",
    "num_birdclef2021_primary_labels",
    "BirdCLEF2021PrimaryLabelDataset",
    "BirdCLEF2021PrimaryLabelComposer",
    "BirdCLEF2021BaselineCollator",
    # for BirdCLEF2022
    "birdclef2022_primary_labels",
    "num_birdclef2022_primary_labels",
    "BirdCLEF2022PrimaryLabelDataset",
    "BirdCLEF2022PrimaryLabelComposer",
    "BirdCLEF2022BaselineCollator",
    # for BirdCLEF2023
    "birdclef2023_primary_labels",
    "num_birdclef2023_primary_labels",
    "BirdCLEF2023PrimaryLabelDataset",
    "BirdCLEF2023PrimaryLabelComposer",
    "BirdCLEF2023BaselineCollator",
    # for BirdCLEF2024
    "birdclef2024_primary_labels",
    "num_birdclef2024_primary_labels",
    "BirdCLEF2024PrimaryLabelDataset",
    "BirdCLEF2024AudioDataset",
    "BirdCLEF2024PrimaryLabelComposer",
    "BirdCLEF2024AudioComposer",
    "BirdCLEF2024BaselineCollator",
]
