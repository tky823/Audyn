import torch
from torch.utils.data.sampler import RandomSampler

from .dataset import GumbelCIFAR100


class GumbelVQVAERandomSampler(RandomSampler):

    def __init__(
        self,
        data_source: GumbelCIFAR100,
        replacement: bool = False,
        num_samples: int = None,
        generator: torch.Generator | None = None,
    ) -> None:
        assert isinstance(
            data_source, GumbelCIFAR100
        ), "Only GumbelCIFAR100 is supported as data_source."

        super().__init__(
            data_source=data_source,
            replacement=replacement,
            num_samples=num_samples,
            generator=generator,
        )

    def set_step(self, step: int) -> None:
        self.data_source: GumbelCIFAR100
        self.data_source.set_step(step)

    def get_step(self) -> int:
        return self.data_source.get_step()
