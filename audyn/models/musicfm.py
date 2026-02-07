import os
import warnings
from typing import Dict, Optional, OrderedDict

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..utils._github import download_file_from_github_release


class MusicFM(nn.Module):
    def __init__(
        self,
        embedding: nn.Module,
        backbone: nn.Module,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "MusicFM":
        """Build pretrained AudioSpectrogramTransformer.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from audyn.models import MusicFM
            >>> model = MusicFM.build_from_pretrained("musicfm_msd")

        .. note::

            Supported pretrained model names are
                - fma
                - musicfm_msd

        """  # noqa: E501
        from ..utils._hydra.utils import instantiate  # to avoid circular import

        pretrained_model_configs = _create_pretrained_model_configs()

        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            pretrained_model_config["_target_"] = f"{cls.__module__}.{cls.__name__}"
            model: MusicFM = instantiate(pretrained_model_config)
            model.load_state_dict(model_state_dict)

            if aggregator is not None:
                model.aggregator = aggregator

            if head is not None:
                model.head = head

            return model
        elif pretrained_model_name_or_path in pretrained_model_configs:
            config = pretrained_model_configs[pretrained_model_name_or_path]
            url = config["url"]
            path = config["path"]
            download_file_from_github_release(url, path=path)
            model = cls.build_from_pretrained(
                path,
                aggregator=aggregator,
                head=head,
            )

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            x = input.unsqueeze(dim=-3)
        elif input.dim() == 4:
            x = input
        else:
            raise ValueError("Only 3D and 4D inputs are supported.")

        x = self.embedding(x)
        output = self.backbone(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output


def _create_pretrained_model_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "musicfm_fma": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.3.0/musicfm_fma.pth",  # noqa: E501
            "path": os.path.join(model_cache_dir, "MusicFM", "6e732c6c", "musicfm_fma.pth"),
            "sha256": "6e732c6c181f4bcf8f7337178d5c576bf9b6e5f930c86b0f36b723a7d3cf3335",
        },
        "musicfm_msd": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.2.0/musicfm_msd.pth",  # noqa: E501
            "path": os.path.join(model_cache_dir, "MusicFM", "4f9c8861", "musicfm_msd.pth"),
            "sha256": "4f9c886171bc4154a558752faeb83ef1147c227579085e192d581d3fd284de1c",
        },
    }

    return pretrained_model_configs
