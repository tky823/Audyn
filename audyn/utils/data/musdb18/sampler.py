from typing import Iterator, List, Optional

from torch.utils.data import RandomSampler

__all__ = [
    "RandomStemsMUSDB18Sampler",
]


class RandomStemsMUSDB18Sampler(RandomSampler):
    def __init__(
        self,
        track_names: List[str],
        replacement: bool = True,
        num_samples: Optional[int] = None,
        generator=None,
    ) -> None:
        from . import sources

        if num_samples is None:
            num_samples = len(sources) * len(track_names)
        else:
            num_samples = len(sources) * num_samples

        super().__init__(
            track_names,
            replacement=replacement,
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
