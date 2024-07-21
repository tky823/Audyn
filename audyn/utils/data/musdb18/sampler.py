from typing import Any, Iterator, List, Optional

from torch.utils.data import RandomSampler, Sampler

__all__ = [
    "RandomStemsMUSDB18Sampler",
]


class RandomStemsMUSDB18Sampler(Sampler):
    """MUSDB18 sampler to generate mixture composed by randomly selected tracks.

    Args:
        track_names (list): Track name list.
        replacement (bool): If ``True``, samples are taken with replacement.
        num_samples (int, optional): Number of sampler per epoch. ``len(track_names)`` is
            used by default.
        generator (torch.Generator, optional): Random number generator.

    .. code-block::

        >>> import torch
        >>> from audyn.utils.data.musdb18.sampler import RandomStemsMUSDB18Sampler
        >>> torch.manual_seed(0)
        >>> track_names = [
        ...     "A Classic Education - NightOwl",
        ...     "Flags - 54",
        ...     "Night Panther - Fire",
        ...     "The Districts - Vermont",
        ...     "Young Griffo - Pennies",
        ... ]
        >>> sampler = RandomStemsMUSDB18Sampler(track_names)
        >>> for indices in sampler:
        ...     print(indices)
        ...
        [1, 0, 4, 3]
        [4, 0, 2, 3]
        [4, 1, 3, 0]
        [1, 4, 0, 0]
        [3, 3, 0, 0]

    """

    def __init__(
        self,
        track_names: List[str],
        replacement: bool = True,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        super().__init__(track_names)

        if replacement:
            self.stem_sampler = _ReplacementRandomStemsSampler(
                track_names,
                num_samples_per_source=num_samples,
                generator=generator,
            )
        else:
            self.stem_sampler = _NoReplacementRandomStemsSampler(
                track_names,
                num_samples_per_source=num_samples,
                generator=generator,
            )

    def __iter__(self) -> Iterator[List[int]]:
        yield from self.stem_sampler

    @property
    def replacement(self) -> bool:
        return self.stem_sampler.replacement

    @property
    def track_names(self) -> List[str]:
        return self.stem_sampler.track_names

    @property
    def num_samples(self) -> int:
        return len(self.track_names)

    @property
    def generator(self) -> Optional[Any]:
        return self.stem_sampler.generator


class _ReplacementRandomStemsSampler(RandomSampler):
    """Core implementation of RandomStemsMUSDB18Sampler with replacement.

    Args:
        track_names (list): Track name list.
        num_samples_per_source (int): Number of samples per source.

    """

    def __init__(
        self,
        track_names: List[str],
        num_samples_per_source: Optional[int] = None,
        generator=None,
    ) -> None:
        from . import sources

        if num_samples_per_source is None:
            num_samples = len(sources) * len(track_names)
        else:
            num_samples = len(sources) * num_samples_per_source

        super().__init__(
            track_names,
            replacement=True,
            num_samples=num_samples,
            generator=generator,
        )

    def __iter__(self) -> Iterator[List[int]]:
        from . import sources

        indices = []

        for idx in super().__iter__():
            indices.append(idx)

            if len(indices) >= len(sources):
                yield indices

                indices = []

    @property
    def track_names(self) -> List[str]:
        return self.data_source


class _NoReplacementRandomStemsSampler(RandomSampler):
    """Core implementation of RandomStemsMUSDB18Sampler without replacement.

    Args:
        track_names (list): Track name list.
        num_samples_per_source (int): Number of samples per source.

    """

    def __init__(
        self,
        track_names: List[str],
        num_samples_per_source: Optional[int] = None,
        generator=None,
    ) -> None:
        if num_samples_per_source is None:
            num_samples = len(track_names)
        else:
            num_samples = num_samples_per_source

        super().__init__(
            track_names,
            replacement=False,
            num_samples=num_samples,
            generator=generator,
        )

    def __iter__(self) -> Iterator[List[int]]:
        from . import sources

        for idx in super().__iter__():
            indices = [idx] * len(sources)

            yield indices

    @property
    def track_names(self) -> List[str]:
        return self.data_source
