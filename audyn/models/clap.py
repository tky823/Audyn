import os
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..modules.clap import PatchEmbedding
from ..utils._github import download_file_from_github_release

__all__ = [
    "LAIONAudioEncoder2023",
]


class LAIONAudioEncoder2023(nn.Module):
    """Audio encoder of LION-CLAP 2023.

    Args:
        embedding (audyn.modules.clap.PatchEmbedding): Patch embedding.
        backbone (nn.Module): Backbone of LAIONAudioEncoder2023, which is typically
            Transformer-based module.

    """

    def __init__(self, embedding: PatchEmbedding, backbone: nn.Module) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone

        assert not embedding.insert_cls_token, "[CLS] token is not supported."
        assert not embedding.insert_dist_token, "[DST] token is not supported."

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of LAIONAudioEncoder2023.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, in_channels, n_bins, n_frames),
                where in_channels corresponds to number of fused chunks.

        Returns:
            torch.Tensor: (batch_size, num_downsampled_patches, embedding_dim), where
                ``num_downsampled_patches`` corresponds to
                ``downsampled_height * downsampled_width``.

        """
        x = self.embedding(input)
        output = self.backbone(x)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
    ) -> "LAIONAudioEncoder2023":
        """Build pretrained LAIONAudioEncoder2023.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.

        Examples:

            >>> from audyn.models.clap import LAIONAudioEncoder2023
            >>> model = LAIONAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")

        .. note::

            Supported pretrained model names are
                - laion-clap-htsat-fused

        """  # noqa: E501
        from ..utils._hydra.utils import instantiate  # to avoid circular import

        pretrained_model_configs = _create_pretrained_model_configs()

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path, map_location=lambda storage, loc: storage
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            pretrained_model_config["_target_"] = f"{cls.__module__}.{cls.__name__}"
            model: LAIONAudioEncoder2023 = instantiate(pretrained_model_config)
            model.load_state_dict(model_state_dict)

            return model
        elif pretrained_model_name_or_path in pretrained_model_configs:
            config = pretrained_model_configs[pretrained_model_name_or_path]
            url = config["url"]
            path = config["path"]
            download_file_from_github_release(url, path=path)
            model = cls.build_from_pretrained(path)

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")


def _create_pretrained_model_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "laion-clap-htsat-fused": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.4/laion-clap-htsat-fused.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "LAIONAudioEncoder2023",
                "laion-clap-htsat-fused.pth",
            ),
        },
    }

    return pretrained_model_configs
