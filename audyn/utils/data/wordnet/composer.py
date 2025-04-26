from typing import Any, Callable, Dict, List, Optional

from ..composer import Composer


class WordNetComposer(Composer):
    """Composer for WordNet.

    Args:
        indexer (callable): Indexer to map name to index.
        keys (list): List of keys.

    Examples:

        >>> import torch
        >>> from audyn.utils.data.wordnet import TrainingMammalDataset, WordNetIndexer, WordNetComposer, WordNetDataLoader
        >>> from audyn.utils.data import Collator
        >>> torch.manual_seed(0)
        >>> indexer = WordNetIndexer.build_from_default_config("mammal")
        >>> composer = WordNetComposer(indexer, keys=["anchor", "positive", "negative"])
        >>> collator = Collator(composer=composer)
        >>> dataset = TrainingMammalDataset()
        >>> dataloader = WordNetDataLoader(
        ...     dataset,
        ...     batch_size=2,
        ...     collate_fn=collator,
        ...     burnin_step=3,
        ...     initial_step=1,
        >>> )
        >>> dataloader.set_epoch(0)
        >>> for batch in dataloader:
        ...     print(batch.keys())
        ...     break
        dict_keys(['negative', 'anchor', 'positive'])

    """  # noqa: E501

    def __init__(
        self,
        indexer: Callable[[str], int],
        keys: Optional[List[str]] = None,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        if keys is None:
            keys = []

        self.indexer = indexer
        self.keys = keys

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        indexer = self.indexer
        keys = self.keys

        sample = super().process(sample)

        for key in keys:
            sample[key] = indexer(sample[key])

        return sample
