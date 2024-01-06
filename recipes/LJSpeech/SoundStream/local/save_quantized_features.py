import functools
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig
from utils.driver import QuantizedFeatureSaver

import audyn
from audyn.utils import instantiate_gan_generator, setup_system
from audyn.utils.data import default_collate_fn
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    dataset = hydra.utils.instantiate(config.train.dataset)

    loader = hydra.utils.instantiate(
        config.train.dataloader,
        dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
        ),
    )

    # generator
    model = instantiate_gan_generator(config.model)
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    feature_saver = QuantizedFeatureSaver(
        loader,
        model,
        config=config,
    )
    feature_saver.run()


def collate_fn(batch: List[Any], data_config: DictConfig) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.

    Returns:
        Dict of batch.

    """
    peak_normalization = data_config.audio.peak_normalization

    dict_batch = default_collate_fn(batch)
    dict_batch["waveform"] = dict_batch["waveform"].unsqueeze(dim=1)

    if peak_normalization:
        amplitude = torch.abs(dict_batch["waveform"])
        vmax, _ = torch.max(amplitude, dim=-1)
        vmax = torch.clamp(vmax, min=1e-8)
        dict_batch["waveform"] = dict_batch["waveform"] / vmax.unsqueeze(dim=-1)

    return dict_batch


if __name__ == "__main__":
    main()
