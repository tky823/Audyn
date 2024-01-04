import functools
from typing import Any, Dict, List

import hydra
import torch
from omegaconf import DictConfig

import audyn
from audyn.models.gan import BaseGAN
from audyn.utils import instantiate_model, setup_system
from audyn.utils.data import default_collate_fn
from audyn.utils.driver import GANGenerator
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_system(config)

    test_dataset = hydra.utils.instantiate(config.test.dataset.test)

    test_loader = hydra.utils.instantiate(
        config.test.dataloader.test,
        test_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            num_stages=config.test.num_stages,
        ),
    )

    generator = instantiate_model(config.model.generator)
    generator = set_device(
        generator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.test.ddp_kwargs,
    )
    discriminator = instantiate_model(config.model.discriminator)
    discriminator = set_device(
        discriminator,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.test.ddp_kwargs,
    )

    model = BaseGAN(generator, discriminator)

    generator = GANGenerator(
        test_loader,
        model,
        config=config,
    )
    generator.run()


def collate_fn(
    batch: List[Any],
    data_config: DictConfig,
    num_stages: int,
) -> Dict[str, Any]:
    """Generate dict-based batch.

    Args:
        batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        num_stages (int): Number of stages for inference in RVQ.

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

    dict_batch["num_stages"] = num_stages

    return dict_batch


if __name__ == "__main__":
    main()
