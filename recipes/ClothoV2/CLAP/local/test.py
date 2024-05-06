from typing import Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig
from utils.driver import RetrievalTester

import audyn
from audyn.utils import instantiate, instantiate_metrics, setup_config
from audyn.utils.data import default_collate_fn
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    test_dataset = instantiate(config.test.dataset.test)
    test_loader = instantiate(
        config.train.dataloader.validation,
        test_dataset,
        collate_fn=collate_fn,
    )

    metrics = instantiate_metrics(config.test.metrics)
    metrics = set_device(metrics, accelerator=config.system.accelerator)

    tester = RetrievalTester(
        test_loader,
        metrics=metrics,
        config=config,
    )
    tester.run()


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    assert len(batch) == 1

    dict_batch = default_collate_fn(batch, keys=keys)

    _ = dict_batch.pop("text")

    # remove additional batch dimensions of text features
    dict_batch["text_embedding"] = dict_batch["text_embedding"].squeeze(dim=0)
    dict_batch["audio_embedding"] = dict_batch["audio_embedding"].squeeze(dim=0)

    return dict_batch


if __name__ == "__main__":
    main()
