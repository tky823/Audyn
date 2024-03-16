from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from utils.models.cascade import PixelCNNVQVAE

from audyn.utils import instantiate_model


def instantiate_cascade_model(
    config: DictConfig,
    pixelcnn_checkpoint: str,
    vqvae_checkpoint: str,
    load_weights: Optional[bool] = None,
) -> PixelCNNVQVAE:
    """Instantiate cascade model of PixelCNN and VQVAE.

    Args:
        config (DictConfig): Config of cascase text-to-wave model.
        pixelcnn_checkpoint (str): Path to pretrained PixelCNN.
        vqvae_checkpoint (str): Path to pretrained VQVAE.
        load_weights (bool, optional): If ``True``, model loads pretrained weights.

    Returns:
        CascadeTextToWave: Constructed cascase text-to-wave model.

    """
    if load_weights is None:
        load_weights = False

    pixelcnn_state_dict = torch.load(
        pixelcnn_checkpoint, map_location=lambda storage, loc: storage
    )
    vqvae_state_dict = torch.load(vqvae_checkpoint, map_location=lambda storage, loc: storage)

    # PixelCNN
    pixelcnn_resolved_config: Dict[str, Any] = pixelcnn_state_dict["resolved_config"]
    pixelcnn_model_config: Dict[str, Any] = pixelcnn_resolved_config["model"]
    pixelcnn_model_config = OmegaConf.create(pixelcnn_model_config)
    pixelcnn: nn.Module = instantiate_model(pixelcnn_model_config)

    # VQVAE
    vqvae_resolved_config: Dict[str, Any] = vqvae_state_dict["resolved_config"]
    vqvae_model_config: Dict[str, Any] = vqvae_resolved_config["model"]
    vqvae_model_config = OmegaConf.create(vqvae_model_config)
    vqvae: nn.Module = instantiate_model(vqvae_model_config)

    model: PixelCNNVQVAE = instantiate_model(
        config,
        pixelcnn=pixelcnn,
        vqvae=vqvae,
    )

    if load_weights:
        model.pixelcnn.load_state_dict(pixelcnn_state_dict["model"])
        model.vqvae.load_state_dict(vqvae_checkpoint["model"])

    return model
