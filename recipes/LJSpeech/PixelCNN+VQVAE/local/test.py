import functools
from typing import Dict, Iterable, List, Optional

import torch
from omegaconf import DictConfig, OmegaConf
from utils.driver import Generator
from utils.utils import instantiate_cascade_model

import audyn
from audyn.utils import instantiate, setup_config
from audyn.utils.data import default_collate_fn, slice_feautures
from audyn.utils.modules import set_device


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    vqvae_state_dict = torch.load(
        config.test.checkpoint.vqvae,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )
    vqvae_config = OmegaConf.create(vqvae_state_dict["resolved_config"])
    codebook_size = config.data.codebook.size
    down_scale = vqvae_config.model.encoder.stride**vqvae_config.model.encoder.num_stacks

    test_dataset = instantiate(config.test.dataset.test)
    test_loader = instantiate(
        config.test.dataloader.test,
        test_dataset,
        collate_fn=functools.partial(
            collate_fn,
            data_config=config.data,
            codebook_size=codebook_size,
            down_scale=down_scale,
        ),
    )

    model = instantiate_cascade_model(
        config.model,
        pixelcnn_checkpoint=config.test.checkpoint.pixelcnn,
        vqvae_checkpoint=config.test.checkpoint.vqvae,
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
    down_scale: int,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Generate dict-based batch.

    Args:
        list_batch (list): Single batch to be collated.
            Type of each data is expected ``Dict[str, torch.Tensor]``.
        data_config (DictConfig): Config of data.
        codebook_size (int): Size of codebook used in VQVAE.
        downscale (int): Scale of downsampling in VQVAE.

    Returns:
        Dict of batch.

    """
    slice_length = data_config.audio.slice_length

    dict_batch = default_collate_fn(list_batch, keys=keys)

    dict_batch = slice_feautures(
        dict_batch,
        slice_length=slice_length,
        key_mapping={
            "melspectrogram": "melspectrogram_slice",
        },
        length_mapping={
            "melspectrogram": "melspectrogram_length",
        },
        random_slice=False,
    )

    batch_size, height, width = dict_batch["melspectrogram_slice"].size()
    factory_kwargs = {
        "dtype": torch.long,
        "device": dict_batch["melspectrogram_slice"].device,
    }
    dict_batch["initial_index"] = torch.randint(
        0,
        codebook_size,
        (batch_size, 1, 1),
        **factory_kwargs,
    )
    dict_batch["height"] = height // down_scale
    dict_batch["width"] = width // down_scale

    return dict_batch


if __name__ == "__main__":
    main()
