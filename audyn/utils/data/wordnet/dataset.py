import os
import warnings
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from ._download import _download_mammal_taxonomy


class TrainingMammalDataset(IterableDataset):
    """Training dataset of WordNet (mammal).

    Args:
        num_neg_samples (int): Number of negative samples.
        length (int, optional): Length of dataset.
        burnin_dampening (float): Dampening parameter for buning in.
        seed (int): Random seed. Default: ``0``.

    Examples:

        >>> import torch
        >>> from audyn.utils.data.wordnet import TrainingMammalDataset
        >>> torch.manual_seed(0)
        >>> dataset = TrainingMammalDataset(num_neg_samples=2, burnin_dampening=0.75)
        >>> dataset.set_burnin(True)
        >>> for sample in dataset:
        ...     print(sample)
        ...     break
        {'anchor': 'leporid.n.01', 'positive': 'lagomorph.n.01', 'negative': ['bear.n.01', 'pony.n.05']}

    """  # noqa: E501

    def __init__(
        self,
        num_neg_samples: int = 1,
        length: Optional[int] = None,
        burnin_dampening: float = 1,
        is_symmetric: bool = False,
        seed: int = 0,
    ) -> None:
        from . import mammal_tags

        super().__init__()

        if is_symmetric:
            raise NotImplementedError("is_symmetric=True is not fully implemented.")

        taxonomy = _download_mammal_taxonomy()

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

        assert tags == mammal_tags

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

    def __iter__(self) -> Iterator[dict[str, Any]]:
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

            if pair["self"] == "mammal.n.01":
                # to avoid empty negative candidates
                anchor = pair["child"]
                positive = pair["self"]
            else:
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
    ) -> "TrainingMammalDataset":
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


class EvaluationMammalDataset(Dataset):
    """Evaluation dataset of WordNet (mammal).

    Examples:

        >>> import torch
        >>> from audyn.utils.data.wordnet import EvaluationMammalDataset
        >>> torch.manual_seed(0)
        >>> dataset = EvaluationMammalDataset()
        >>> for sample in dataset:
        ...     print(sample["anchor"])
        ...     print(sample["positive"])
        ...     print(sample["negative"][:5])
        ...     break
        aardvark.n.01
        ['mammal.n.01', 'placental.n.01']
        ['aardwolf.n.01', 'aberdeen_angus.n.01', 'abrocome.n.01', 'abyssinian.n.01', 'addax.n.01']

    """  # noqa: E501

    def __init__(
        self,
        is_symmetric: bool = False,
    ) -> None:
        from . import mammal_tags

        super().__init__()

        taxonomy = _download_mammal_taxonomy()
        tags = []

        for sample in taxonomy:
            name = sample["name"]
            tags.append(name)

        assert tags == mammal_tags

        self.tags = tags
        self.taxonomy = taxonomy

        self.is_symmetric = is_symmetric

    def __getitem__(self, index: int) -> dict[str, Any]:
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
    ) -> "EvaluationMammalDataset":
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
