import functools
from typing import Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from utils.driver import Generator
from utils.utils import instantiate_cascade_model

import audyn
from audyn.utils import instantiate, setup_config
from audyn.utils.data import default_collate_fn
from audyn.utils.model import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    vqvae_state_dict = torch.load(
        config.test.checkpoint.vqvae, map_location=lambda storage, loc: storage
    )
    vqvae_config = OmegaConf.create(vqvae_state_dict["resolved_config"])
    codebook_size = vqvae_config.data.codebook.size

    test_dataset = instantiate(config.test.dataset.test)
    test_loader = instantiate(
        config.test.dataloader.test,
        test_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
        ),
    )

    model = instantiate_cascade_model(
        config.model,
        pixelsnail_checkpoint=config.test.checkpoint.pixelsnail,
        vqvae_checkpoint=config.test.checkpoint.vqvae,
        hifigan_checkpoint=config.test.checkpoint.hifigan,
    )
    model = set_device(
        model,
        accelerator=config.system.accelerator,
        is_distributed=config.system.distributed.enable,
        ddp_kwargs=config.train.ddp_kwargs,
    )

    generator = Generator(
        test_loader,
        model,
        config=config,
    )
    generator.run()


def collate_fn(
    list_batch: List[Dict[str, torch.Tensor]],
    data_config: DictConfig,
    codebook_size: int,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        codebook_size (int): Size of codebook used in VQVAE.

    Returns:
        Dict of batch.

    """
    codebook_height, codebook_width = data_config.codebook.shape

    dict_batch = default_collate_fn(list_batch, keys=keys)

    dict_batch["initial_index"] = torch.randint(
        0, codebook_size, (len(dict_batch["filename"]), 1, 1), dtype=torch.long
    )
    dict_batch["height"] = codebook_height
    dict_batch["width"] = codebook_width

    return dict_batch


if __name__ == "__main__":
    main()
