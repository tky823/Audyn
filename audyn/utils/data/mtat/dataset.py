from torch.utils.data import Dataset

from ._download import download_annotations, download_tags


class MTATDataset(Dataset):
    """Dataset for MagnaTagATune."""

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
    ) -> None:
        super().__init__()

        filenames = []

        with open(list_path) as f:
            for line in f:
                filenames.append(line.strip("\n"))

        self.feature_dir = feature_dir

        for filename in filenames:
            pass
