import os
from collections import OrderedDict
from typing import Callable, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from packaging import version
from torch.nn.common_types import _size_1_t
from torch.nn.modules.utils import _single

from ..modules.transformer import get_activation
from ..utils._github import download_file_from_github_release

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")

__all__ = [
    "HuBERT",
    "HuBERTEmbedding",
    "HuBERTEncoder",
]


class BaseHuBERT(nn.Module):
    """Base class of hidden unit BERT (HuBERT)."""

    def __init__(
        self,
        embedding: "HuBERTEmbedding",
        backbone: "HuBERTEncoder",
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of BaseHuBERT.
        Args:
            input (torch.Tensor): Waveform of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Extracted feature of shape (batch_size, num_frames, embedding_dim).

        """
        # TODO: support padding mask
        x = self.embedding(input)
        output = self.backbone(x)

        return output

    @classmethod
    def build_from_pretrained(cls, pretrained_model_name_or_path: str) -> "BaseHuBERT":
        """Build HuBERT from pretrained model.

        Args:
            pretrained_model_name_or_path (str): Pretrained model name or path.

        Returns:
            BaseHuBERT: Built HuBERT.

        """
        from ..utils._hydra.utils import instantiate  # to avoid circular import

        pretrained_model_configs = _create_pretrained_configs()

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
            model: BaseHuBERT = instantiate(pretrained_model_config)
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


class HuBERT(BaseHuBERT):
    pass


class HuBERTEmbedding(nn.Module):
    """HuBERT embedding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_features: Union[int, list[int]],
        kernel_size: list[_size_1_t],
        stride: list[_size_1_t],
        activation: str = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        num_blocks = len(kernel_size)

        if isinstance(num_features, list):
            assert len(num_features) == num_blocks
        else:
            num_features = [num_features] * num_blocks

        assert len(stride) == num_blocks

        backbone = []

        for block_idx in range(num_blocks):
            if block_idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = num_features[block_idx - 1]

            _out_channels = num_features[block_idx]
            _kernel_size = kernel_size[block_idx]
            _stride = stride[block_idx]

            backbone.append(
                HuBERTEmbeddingBlock(
                    _in_channels,
                    _out_channels,
                    kernel_size=_kernel_size,
                    stride=_stride,
                    activation=activation,
                )
            )

        self.backbone = nn.ModuleList(backbone)
        self.norm = nn.LayerNorm(num_features[-1])
        self.linear = nn.Linear(num_features[-1], out_channels)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input

        for block in self.backbone:
            x = block(x)

        x = x.transpose(-2, -1)
        x = self.norm(x)
        x = self.linear(x)
        output = self.dropout(x)

        return output


class HuBERTEncoder(nn.Module):
    """HuBERT encoder."""

    def __init__(
        self, embedding: "HuBERTEncoderPositionalEmbedding", backbone: nn.TransformerEncoder
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input)
        output = self.backbone(x)

        return output


class HuBERTEmbeddingBlock(nn.Module):
    """HuBERT embedding block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 5,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.norm1d = nn.LayerNorm((out_channels,))
        self.activation1d = get_activation(activation)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.conv1d(input)
        x = x.transpose(-2, -1)
        x = self.norm1d(x)
        x = x.transpose(-2, -1)
        output = self.activation1d(x)

        return output


class HuBERTEncoderPositionalEmbedding(nn.Module):
    """Positional embedding for HuBERT encoder."""

    def __init__(
        self,
        embedding_dim: int,
        kernel_size: _size_1_t,
        groups: int = 1,
        norm: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.gelu,
        dropout: float = 0.1,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        (kernel_size,) = _single(kernel_size)

        self.norm = norm
        self.conv1d = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size,
            stride=1,
            groups=groups,
        )
        self.activation = get_activation(activation)
        self.dropout = nn.Dropout(p=dropout)

        self.registered_weight_norms = set()

        if weight_regularization is None:
            pass
        elif weight_regularization == "weight_norm":
            self.weight_norm_()
        else:
            raise ValueError(
                f"{weight_regularization}-based weight regularization is not supported."
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of HuBERTEncoderPositionalEmbedding.

        Args:
            input (torch.Tensor): Extracted feature of shape (batch_size, length, embedding_dim).

        Returns:
            torch.Tensor: Embedding of shape (batch_size, length, embedding_dim).

        """
        (kernel_size,) = self.conv1d.kernel_size

        x = input.transpose(-2, -1)

        if self.norm is not None:
            x = self.norm(x)

        padding = kernel_size - 1
        padding_right = padding // 2
        padding_left = padding - padding_right

        x = F.pad(x, (padding_left, padding_right))
        x = self.conv1d(x)
        x = self.activation(x)
        x = input + x.transpose(-2, -1)
        output = self.dropout(x)

        return output

    def weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        weight_norm_kwargs = {"dim": 2}

        if "conv1d" not in self.registered_weight_norms:
            self.conv1d = weight_norm_fn(self.conv1d, **weight_norm_kwargs)
            self.registered_weight_norms.add("conv1d")

    def remove_weight_norm_(self) -> None:
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        if "conv1d" in self.registered_weight_norms:
            self.conv1d = remove_weight_norm_fn(self.conv1d, *remove_weight_norm_args)
            self.registered_weight_norms.pop("conv1d")


def _create_pretrained_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "hubert-large-librispeech960-finetuning": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.6/hubert-large-librispeech960-finetuning.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "HuBERT",
                "f25ba5bc",
                "hubert-large-librispeech960-finetuning.pth",
            ),
            "sha256": "f25ba5bcaf4a081a5163f2f1dd4d50df5e4f87bd416f415d140f0239eb073e0d",
        },
    }

    return pretrained_model_configs
