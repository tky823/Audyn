from torch.utils.data import Dataset


class MTGJamendoDataset(Dataset):

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        task: str,
    ) -> None:
        super().__init__()

        from . import supported_tasks

        assert task in supported_tasks

    def __getitem__(self, idx: int) -> Any:
        return super().__getitem__(index)

    def __len__(self) -> int:
        return len(self.samples)
