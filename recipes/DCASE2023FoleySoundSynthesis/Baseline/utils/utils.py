from typing import Any, Dict, Optional

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from utils.models.cascade import BaselineModel


def instantiate_cascade_model(
    config: DictConfig,
    pixelsnail_checkpoint: str,
    vqvae_checkpoint: str,
    hifigan_checkpoint: str,
    load_weights: Optional[bool] = None,
) -> BaselineModel:
    """Instantiate cascade of PixelSNAIL, VQVAE, and HiFi-GAN.

    Args:
        config (DictConfig): Config of cascase text-to-wave model.
        pixelsnail_checkpoint (str): Path to pretrained PixelSNAIL.
        vqvae_checkpoint (str): Path to pretrained VQVAE.
        hifigan_checkpoint (str): Path to pretrained HiFi-GAN.
        load_weights (bool, optional): If ``True``, model loads pretrained weights.

    Returns:
        CascadeTextToWave: Constructed cascase text-to-wave model.

    """
    if load_weights is None:
        load_weights = False

    pixelsnail_state_dict = torch.load(
        pixelsnail_checkpoint, map_location=lambda storage, loc: storage
    )
    vqvae_state_dict = torch.load(vqvae_checkpoint, map_location=lambda storage, loc: storage)
    hifigan_state_dict = torch.load(hifigan_checkpoint, map_location=lambda storage, loc: storage)

    # PixelSNAIL
    pixelsnail_resolved_config: Dict[str, Any] = pixelsnail_state_dict["resolved_config"]
    pixelsnail_model_config: Dict[str, Any] = pixelsnail_resolved_config["model"]
    pixelsnail_model_config = OmegaConf.create(pixelsnail_model_config)
    pixelsnail: nn.Module = hydra.utils.instantiate(pixelsnail_model_config)

    # VQVAE
    vqvae_resolved_config: Dict[str, Any] = vqvae_state_dict["resolved_config"]
    vqvae_model_config: Dict[str, Any] = vqvae_resolved_config["model"]
    vqvae_model_config = OmegaConf.create(vqvae_model_config)
    vqvae: nn.Module = hydra.utils.instantiate(vqvae_model_config)

    # HiFi-GAN
    hifigan_resolved_config: Dict[str, Any] = hifigan_state_dict["resolved_config"]
    hifigan_model_config: Dict[str, Any] = hifigan_resolved_config["model"]
    hifigan_model_config = OmegaConf.create(hifigan_model_config)
    hifigan_generator: nn.Module = hydra.utils.instantiate(hifigan_model_config.generator)

    model: BaselineModel = hydra.utils.instantiate(
        config,
        pixelsnail=pixelsnail,
        vqvae=vqvae,
        hifigan_generator=hifigan_generator,
    )

    if load_weights:
        model.pixelsnail.load_state_dict(pixelsnail_state_dict["model"])
        model.vqvae.load_state_dict(vqvae_checkpoint["model"])
        model.hifigan_generator.load_state_dict(hifigan_checkpoint["model"])

    return model
