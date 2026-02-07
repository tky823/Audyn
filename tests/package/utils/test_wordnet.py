import torch
from torch.utils.data import DataLoader

from audyn.utils.data import Collator
from audyn.utils.data.wordnet.composer import WordNetComposer
from audyn.utils.data.wordnet.dataloader import WordNetDataLoader
from audyn.utils.data.wordnet.dataset import (
    EvaluationMammalDataset,
    TrainingMammalDataset,
)
from audyn.utils.data.wordnet.indexer import WordNetIndexer


def test_wordnet_dataloader() -> None:
    torch.manual_seed(0)

    batch_size = 2
    burnin_step = 5

    indexer = WordNetIndexer.build_from_default_config("mammal")
    composer = WordNetComposer(indexer, keys=["anchor", "positive", "negative"])
    collator = Collator(composer=composer)
    training_dataset = TrainingMammalDataset()
    evaluation_dataset = EvaluationMammalDataset()
    training_dataloader = WordNetDataLoader(
        training_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        burnin_step=burnin_step,
    )
    evaluation_dataloader = DataLoader(
        evaluation_dataset,
        batch_size=1,
        collate_fn=collator,
    )

    for epoch_idx in range(3):
        training_dataloader.set_epoch(epoch_idx)

        for batch in training_dataloader:
            assert set(batch.keys()) == {"anchor", "positive", "negative"}
            break

    for batch in evaluation_dataloader:
        assert set(batch.keys()) == {"anchor", "positive", "negative"}
        break

    name = indexer.decode(0)
    names_by_scaler = indexer.decode([0, 1])

    assert name == "aardvark.n.01"
    assert names_by_scaler == ["aardvark.n.01", "aardwolf.n.01"]
