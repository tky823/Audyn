import os
import warnings
from typing import Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ._download import download_audioset_taxonomy


class TrainingAudioSetDataset(IterableDataset):
    """Training dataset of AudioSet for tag embedding.

    Args:
        num_neg_samples (int): Number of negative samples.
        length (int, optional): Length of dataset.
        burnin_dampening (float): Dampening parameter for buning in.
        seed (int): Random seed. Default: ``0``.

    Examples:

        >>> import torch
        >>> from utils.dataset import TrainingAudioSetDataset
        >>> torch.manual_seed(0)
        >>> dataset = TrainingAudioSetDataset(num_neg_samples=2, burnin_dampening=0.75)
        >>> dataset.set_burnin(True)
        >>> for sample in dataset:
        ...     print(sample)
        ...     break
        {'anchor': 'Cat communication', 'positive': 'Cat', 'negative': ['Arrow', 'Non-motorized land vehicle']}

    """  # noqa: E501

    def __init__(
        self,
        num_neg_samples: int = 1,
        length: int | None = None,
        burnin_dampening: float = 1,
        is_symmetric: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()

        taxonomy = download_audioset_taxonomy()

        tags = []
        pair_list = []
        weights = {}

        for sample in taxonomy:
            name = sample["name"]
            tags.append(name)

            if name not in weights:
                weights[name] = 0

            for child_name in sample["child"]:
                pair_list.append({"self": name, "child": child_name})

                weights[name] += 1

                if child_name not in weights:
                    weights[child_name] = 0

                if is_symmetric:
                    weights[child_name] += 1

        if length is None:
            length = len(pair_list)

        self.tags = tags
        self.taxonomy = taxonomy
        self.pair_list = pair_list

        self.num_neg_samples = num_neg_samples
        self.length = length
        self.burnin = None
        self.burnin_dampening = burnin_dampening
        self.weights = weights

        self.is_symmetric = is_symmetric

        self.generator = None
        self.seed = seed
        self.epoch_index = 0  # to share random state among workers

    def __iter__(self) -> Iterator[tuple[str, str, list[str]]]:
        tags = self.tags
        taxonomy = self.taxonomy
        pair_list = self.pair_list
        num_neg_samples = self.num_neg_samples
        length = self.length
        burnin = self.burnin
        is_symmetric = self.is_symmetric
        seed = self.seed
        epoch_index = self.epoch_index

        worker_info = get_worker_info()
        is_distributed = dist.is_available() and dist.is_initialized()

        if worker_info is None:
            local_worker_id = 0
            num_local_workers = 1
        else:
            local_worker_id = worker_info.id
            num_local_workers = worker_info.num_workers

        if is_distributed:
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1

        global_worker_id = rank * num_local_workers + local_worker_id
        num_global_workers = world_size * num_local_workers

        if burnin is None:
            raise ValueError(
                "Set burnin by calling set_burnin(True) or set_burnin(False) before iteration."
            )

        if self.generator is None:
            self.generator = torch.Generator()

        # share random state among workers to random sampling
        self.generator.manual_seed(seed + epoch_index)

        indices = torch.randint(
            0,
            len(pair_list),
            (length,),
            generator=self.generator,
        )
        indices = indices.tolist()
        indices = indices[global_worker_id::num_global_workers]

        for pair_index in indices:
            pair = pair_list[pair_index]

            if is_symmetric:
                if torch.rand((), generator=self.generator) < 0.5:
                    anchor = pair["self"]
                    positive = pair["child"]
                else:
                    anchor = pair["child"]
                    positive = pair["self"]
            else:
                anchor = pair["child"]
                positive = pair["self"]

            anchor_index = tags.index(anchor)
            parent = taxonomy[anchor_index]["parent"]
            child = taxonomy[anchor_index]["child"]

            if is_symmetric:
                positive_candidates = set(parent) | set(child)
            else:
                positive_candidates = set(parent)

            negative_candidates = set(tags) - positive_candidates - {anchor}
            negative_candidates = sorted(list(negative_candidates))

            if burnin:
                negative = self.sample(
                    negative_candidates,
                    num_samples=num_neg_samples,
                    weights=self.weights,
                    dampening=self.burnin_dampening,
                    replacement=False,
                    generator=self.generator,
                )
            else:
                negative_indices = torch.randperm(
                    len(negative_candidates), generator=self.generator
                )
                negative_indices = negative_indices[:num_neg_samples]
                negative_indices = negative_indices.tolist()

                negative = []

                for negative_index in negative_indices:
                    _negative = negative_candidates[negative_index]
                    negative.append(_negative)

            sample = {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
            }

            yield sample

        self.epoch_index = epoch_index + 1

    def set_burnin(self, burnin: bool) -> None:
        self.burnin = burnin

    @classmethod
    def build_from_list_path(
        cls,
        list_path: str,
        feature_dir: str,
        num_neg_samples: int = 1,
        length: Optional[int] = None,
        burnin_dampening: float = 1,
        is_symmetric: bool = False,
        seed: int = 0,
    ) -> "TrainingAudioSetDataset":
        dataset = cls(
            num_neg_samples=num_neg_samples,
            length=length,
            burnin_dampening=burnin_dampening,
            is_symmetric=is_symmetric,
            seed=seed,
        )

        num_samples = 0

        with open(list_path) as f:
            for _ in f:
                num_samples += 1

        assert len(dataset.taxonomy) == num_samples

        if os.path.exists(feature_dir) and len(os.listdir(feature_dir)) > 0:
            warnings.warn(f"{feature_dir} exists, but is not used.", UserWarning, stacklevel=2)

        return dataset

    @staticmethod
    def sample(
        candidates: list[str],
        num_samples: int = 1,
        weights: Optional[dict[str, float]] = None,
        dampening: float = 1,
        replacement: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> list[str]:
        """Sample from candidates based on weights.

        Args:
            weights (dict): Dictionary that maps candidate to weight.
            dampening (float): Dampening parameter for negative sampling.

        Returns:
            list: List of sampled candidates.

        """
        candidate_weights = []

        for candidate in candidates:
            if weights is None:
                weight = 1
            else:
                weight = weights[candidate]

            candidate_weights.append(weight**dampening)

        candidate_weights = torch.tensor(candidate_weights, dtype=torch.float)
        indices = torch.multinomial(
            candidate_weights,
            num_samples,
            replacement=replacement,
            generator=generator,
        )
        indices = indices.tolist()

        samples = []

        for index in indices:
            _sample = candidates[index]
            samples.append(_sample)

        return samples

    def __len__(self) -> int:
        return self.length


class EvaluationAudioSetDataset(Dataset):
    """Evaluation dataset of AudioSet for tag embedding.

    Examples:

        >>> import torch
        >>> from utils.dataset import EvaluationAudioSetDataset
        >>> torch.manual_seed(0)
        >>> dataset = EvaluationAudioSetDataset()
        >>> for sample in dataset:
        ...     print(sample["anchor"])
        ...     print(sample["positive"])
        ...     print(sample["negative"][:5])
        ...     break
        A capella
        ['Vocal music']
        ['Accelerating, revving, vroom', 'Accordion', 'Acoustic environment', 'Acoustic guitar', 'Afrobeat']

    """  # noqa: E501

    def __init__(
        self,
        is_symmetric: bool = False,
    ) -> None:
        super().__init__()

        taxonomy = download_audioset_taxonomy()

        tags = []

        for sample in taxonomy:
            name = sample["name"]
            tags.append(name)

        self.tags = tags
        self.taxonomy = taxonomy

        self.is_symmetric = is_symmetric

    def __getitem__(self, index: int) -> tuple[str, str, list[str]]:
        tags = self.tags
        taxonomy = self.taxonomy
        is_symmetric = self.is_symmetric

        anchor_index = index
        anchor = taxonomy[anchor_index]["name"]
        parent = taxonomy[anchor_index]["parent"]
        child = taxonomy[anchor_index]["child"]

        if is_symmetric:
            positive_candidates = set(parent) | set(child)
        else:
            positive_candidates = set(parent)

        negative_candidates = set(tags) - set(positive_candidates) - {anchor}

        positive = sorted(list(positive_candidates))
        negative = sorted(list(negative_candidates))

        sample = {
            "anchor": anchor,
            "positive": positive,
            "negative": negative,
        }

        return sample

    @classmethod
    def build_from_list_path(
        cls,
        list_path: str,
        feature_dir: str,
        is_symmetric: bool = False,
    ) -> "EvaluationAudioSetDataset":
        dataset = cls(
            is_symmetric=is_symmetric,
        )

        num_samples = 0

        with open(list_path) as f:
            for _ in f:
                num_samples += 1

        assert len(dataset.taxonomy) == num_samples

        if os.path.exists(feature_dir) and len(os.listdir(feature_dir)) > 0:
            warnings.warn(f"{feature_dir} exists, but is not used.", UserWarning, stacklevel=2)

        return dataset

    def __len__(self) -> int:
        return len(self.taxonomy)
