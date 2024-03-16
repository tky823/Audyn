from typing import Any, Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig
from utils.driver import PriorSaver

import audyn
from audyn.utils import instantiate, instantiate_model, setup_system
from audyn.utils.data import default_collate_fn, take_log_features
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    dataset = instantiate(config.train.dataset)
    loader = instantiate(
        config.train.dataloader,
        dataset,
        collate_fn=collate_fn,
    )

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    saver = PriorSaver(loader, model, config=config)
    saver.run()


def collate_fn(
    list_batch: List[Dict[str, Any]],
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(list_batch, keys=keys)
    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
        },
    )

    return dict_batch


if __name__ == "__main__":
    main()
