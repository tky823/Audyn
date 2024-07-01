from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from utils.driver import EmbeddingSaver

import audyn
from audyn.utils import instantiate, instantiate_model, setup_config
from audyn.utils.data import default_collate_fn, take_log_features
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    test_dataset = instantiate(config.test.dataset.test)
    test_loader = instantiate(
        config.train.dataloader.validation,
        test_dataset,
        collate_fn=collate_fn,
    )

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    saver = EmbeddingSaver(test_loader, model, config=config)
    saver.run()


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

    sample_idx = 0
    sample = batch[sample_idx]

    tokens = sample.pop("tokens")
    tokens, tokens_length = nn.utils.rnn.pad_packed_sequence(tokens, batch_first=True)
    sample["text"] = tokens
    sample["text_length"] = tokens_length

    dict_batch = default_collate_fn(batch, keys=keys)

    # remove additional batch dimensions of text features
    dict_batch["text"] = dict_batch["text"].squeeze(dim=0)
    dict_batch["text_length"] = dict_batch["text_length"].squeeze(dim=0)

    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
        },
        flooring_fn=lambda x: torch.clamp(x, min=1e-10),
    )

    return dict_batch


if __name__ == "__main__":
    main()
