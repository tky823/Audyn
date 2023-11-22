import functools
from typing import Any, Dict, Iterable, List, Optional

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from utils.driver import PriorSaver

import audyn
from audyn.utils import instantiate_model, setup_system
from audyn.utils.data import default_collate_fn, take_log_features
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    down_scale = config.model.encoder.stride**config.model.encoder.num_stacks
    n_frames = config.data.audio.length // config.data.melspectrogram.hop_length
    n_frames = n_frames - (n_frames % down_scale)

    dataset = hydra.utils.instantiate(config.train.dataset)
    loader = hydra.utils.instantiate(
        config.train.dataloader,
        dataset,
        collate_fn=functools.partial(collate_fn, n_frames=n_frames),
    )

    model = instantiate_model(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
    )

    saver = PriorSaver(loader, model, config=config)
    saver.run()


def collate_fn(
    list_batch: List[Dict[str, Any]],
    n_frames: int,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        n_frames (int): Number of frames in Mel-spectrogram.
        keys (iterable, optional): Keys to generate batch.
            If ``None`` is given, all keys detected in ``batch`` are used.
            Default: ``None``.

    Returns:
        Dict of batch.

    """
    dict_batch = default_collate_fn(list_batch, keys=keys)
    n_frames_in = dict_batch["melspectrogram"].size(-1)
    dict_batch["melspectrogram"] = F.pad(dict_batch["melspectrogram"], (0, n_frames - n_frames_in))
    dict_batch = take_log_features(
        dict_batch,
        key_mapping={
            "melspectrogram": "log_melspectrogram",
        },
    )

    return dict_batch


if __name__ == "__main__":
    main()
