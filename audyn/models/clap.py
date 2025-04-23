import os
import warnings
from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from ..modules.clap import PatchEmbedding
from ..utils._github import download_file_from_github_release
from .ast import Aggregator, Head

__all__ = [
    "LAIONCLAPAudioEncoder2023",
    "LAIONAudioEncoder2023",
    "MicrosoftCLAPAudioEncoder2023",
    "MicrosoftAudioEncoder2023",
]


class LAIONCLAPAudioEncoder2023(nn.Module):
    """Audio encoder of LAION-CLAP 2023.

    Args:
        embedding (audyn.modules.clap.PatchEmbedding): Patch embedding.
        backbone (nn.Module): Backbone of LAIONCLAPAudioEncoder2023, which is typically
            Transformer-based module.
        aggregator (nn.Module, optional): Aggregator.
        head (nn.Module, optional): Head module to project output features.

    Examples:

        >>> import torch
        >>> from audyn.transforms import LAIONCLAPAudioEncoder2023WaveformPad, LAIONCLAPAudioEncoder2023MelSpectrogram, LAIONCLAPAudioEncoder2023MelSpectrogramFusion
        >>> from audyn.models import LAIONCLAPAudioEncoder2023
        >>> torch.manual_seed(0)
        >>> waveform_padding = LAIONCLAPAudioEncoder2023WaveformPad.build_from_pretrained("laion-clap-htsat-fused")
        >>> melspectrogram_transform = LAIONCLAPAudioEncoder2023MelSpectrogram.build_from_pretrained("laion-clap-htsat-fused")
        >>> fusion_transform = LAIONCLAPAudioEncoder2023MelSpectrogramFusion.build_from_pretrained("laion-clap-htsat-fused")
        >>> model = LAIONCLAPAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")
        >>> batch_size = 2
        >>> sample_rate = melspectrogram_transform.sample_rate
        >>> print(sample_rate)
        48000
        # short waveform
        >>> length = int(0.4 * waveform_padding.min_length)
        >>> waveform = torch.randn((batch_size, length))
        >>> print(waveform.size())
        torch.Size([2, 192000])
        >>> waveform = waveform_padding(waveform)
        >>> print(waveform.size())
        torch.Size([2, 480000])
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> print(melspectrogram.size())
        torch.Size([2, 64, 1001])
        >>> fused_melspectrogram = fusion_transform(melspectrogram)
        >>> print(fused_melspectrogram.size())
        torch.Size([2, 4, 64, 1001])
        >>> embedding = model(fused_melspectrogram)
        >>> print(embedding.size())
        torch.Size([2, 512])
        # long waveform
        >>> length = int(1.2 * waveform_padding.min_length)
        >>> waveform = torch.randn((batch_size, length))
        >>> print(waveform.size())
        torch.Size([2, 576000])
        >>> waveform = waveform_padding(waveform)
        >>> print(waveform.size())
        torch.Size([2, 576000])
        >>> melspectrogram = melspectrogram_transform(waveform)
        >>> print(melspectrogram.size())
        torch.Size([2, 64, 1201])
        >>> fused_melspectrogram = fusion_transform(melspectrogram)
        >>> print(fused_melspectrogram.size())
        torch.Size([2, 4, 64, 1001])
        >>> embedding = model(fused_melspectrogram)
        >>> print(embedding.size())
        torch.Size([2, 512])

    .. note::

        Normalization is not applied to embedding. To normalize, use
        ``F.normalize(embedding, p=2, dim=-1)``.

    """  # noqa: E501

    def __init__(
        self,
        embedding: PatchEmbedding,
        backbone: nn.Module,
        aggregator: Optional[Aggregator] = None,
        head: Optional[Head] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        assert not embedding.insert_cls_token, "[CLS] token is not supported."
        assert not embedding.insert_dist_token, "[DST] token is not supported."

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of LAIONCLAPAudioEncoder2023.

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

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "LAIONCLAPAudioEncoder2023":
        """Build pretrained LAIONCLAPAudioEncoder2023.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from audyn.models.clap import LAIONCLAPAudioEncoder2023
            >>> model = LAIONCLAPAudioEncoder2023.build_from_pretrained("laion-clap-htsat-fused")

        .. note::

            Supported pretrained model names are
                - laion-clap-htsat-fused

        """  # noqa: E501
        from ..utils._hydra.utils import instantiate  # to avoid circular import

        pretrained_model_configs = _create_pretrained_laion_clap_configs()

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
            model: LAIONCLAPAudioEncoder2023 = instantiate(pretrained_model_config)
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
            model = cls.build_from_pretrained(path, aggregator=aggregator, head=head)

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")


class MicrosoftCLAPAudioEncoder2023(nn.Module):
    """Audio encoder of Microsoft-CLAP 2023.

    Args:
        embedding (audyn.modules.clap.PatchEmbedding): Patch embedding.
        backbone (nn.Module): Backbone of MicrosoftCLAPAudioEncoder2023, which is typically
            Transformer-based module.
        aggregator (nn.Module, optional): Aggregator.
        head (nn.Module, optional): Head module to project output features.

    .. note::

        Normalization is not applied to embedding. To normalize, use
        ``F.normalize(embedding, p=2, dim=-1)``.

    """  # noqa: E501

    def __init__(
        self,
        embedding: PatchEmbedding,
        backbone: nn.Module,
        aggregator: Optional[Aggregator] = None,
        head: Optional[Head] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        assert not embedding.insert_cls_token, "[CLS] token is not supported."
        assert not embedding.insert_dist_token, "[DST] token is not supported."

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MicrosoftCLAPAudioEncoder2023.

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

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "MicrosoftCLAPAudioEncoder2023":
        """Build pretrained MicrosoftCLAPAudioEncoder2023.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from audyn.models.clap import MicrosoftCLAPAudioEncoder2023
            >>> model = MicrosoftCLAPAudioEncoder2023.build_from_pretrained("microsoft-clap-2023")

        .. note::

            Supported pretrained model names are
                - microsoft-clap-2023

        """  # noqa: E501
        from ..utils._hydra.utils import instantiate  # to avoid circular import

        pretrained_model_configs = _create_pretrained_microsoft_clap_configs()

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
            model: MicrosoftCLAPAudioEncoder2023 = instantiate(pretrained_model_config)
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
            model = cls.build_from_pretrained(path, aggregator=aggregator, head=head)

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")


class LAIONAudioEncoder2023(LAIONCLAPAudioEncoder2023):
    """Alias of LAIONCLAPAudioEncoder2023."""


class MicrosoftAudioEncoder2023(MicrosoftCLAPAudioEncoder2023):
    """Alias of MicrosoftCLAPAudioEncoder2023."""


class MLPHead(Head):
    """Projection for CLAP audio feature."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPHead.

        Args:
            input (torch.Tensor): Aggregated feature of shape (batch_size, in_channels).

        Returns:
            torch.Tensor: Transformed feature of shape (batch_size, out_channels).

        """
        x = self.linear1(input)
        x = self.activation(x)
        output = self.linear2(x)

        return output


class LAIONMLPHead(MLPHead):
    """Alias of MLPHead."""


class MicrosoftMLPHead(Head):
    """Projection for Microsoft-CLAP audio feature."""

    def __init__(
        self, in_channels: int, out_channels: int, bias: bool = False, dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(in_channels, out_channels, bias=bias)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(out_channels, out_channels, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPHead.

        Args:
            input (torch.Tensor): Aggregated feature of shape (batch_size, in_channels).

        Returns:
            torch.Tensor: Transformed feature of shape (batch_size, out_channels).

        """
        x_1 = self.linear1(input)
        x_2 = self.activation(x_1)
        x_2 = self.linear2(x_2)
        x_2 = self.dropout(x_2)
        output = self.norm(x_1 + x_2)

        return output


def _create_pretrained_laion_clap_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "laion-clap-htsat-fused": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.5/laion-clap-htsat-fused.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "LAIONCLAPAudioEncoder2023",
                "138f4a83",
                "laion-clap-htsat-fused.pth",
            ),
            "sha256": "138f4a83b2b68d799fbef5f7af4937ec13e86b1e8f3964a6e13407376a4fe1d4",
        },
    }

    return pretrained_model_configs


def _create_pretrained_microsoft_clap_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "microsoft-clap-2023": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.5/microsoft-clap-2023.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "MicrosoftCLAPAudioEncoder2023",
                "589ce69e",
                "microsoft-clap-2023.pth",
            ),
            "sha256": "589ce69ebbc6e7af3afc3d998edbfa0752fd54b0781e35cf98bffd9b626b3dfa",
        },
    }

    return pretrained_model_configs
