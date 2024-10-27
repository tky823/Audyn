import os
import warnings
from typing import Dict, Optional, OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from ..modules.music_tagging_transformer import (
    MusicTaggingTransformerEncoder,
    PositionalPatchEmbedding,
)
from ..utils.github import download_file_from_github_release
from .ast import BaseAudioSpectrogramTransformer, HeadTokensAggregator, MLPHead

__all__ = [
    "MusicTaggingTransformer",
    "MusicTaggingTransformerLinearProbing",
]


class MusicTaggingTransformer(BaseAudioSpectrogramTransformer):
    """Music Tagging Transformer proposed by [#won2021semi]_.

    .. [#won2021semi]
        M. Won et al., "Semi-supervised music tagging transformer,"
        *arXiv preprint arXiv:2111.13457*.

    """

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: MusicTaggingTransformerEncoder,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(embedding=embedding, backbone=backbone)

        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior.",
                UserWarning,
                stacklevel=2,
            )

    def forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        input = self.pad_by_length(input, length=length)
        x = self.embedding(input)
        padding_mask = self.compute_padding_mask(input, length=length)
        output = self.transformer_forward(x, padding_mask=padding_mask)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output

    def compute_padding_mask(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.LongTensor:
        if length is None:
            padding_mask = None
        else:
            factory_kwargs = {
                "dtype": torch.long,
                "device": length.device,
            }
            _, n_bins, max_frames = input.size()
            width = []

            for _length in length:
                n_frames = _length.item()
                _width = self.embedding.compute_output_shape(n_bins, n_frames)
                width.append(_width)

            width = torch.tensor(width, **factory_kwargs)
            max_width = self.embedding.compute_output_shape(n_bins, max_frames)
            padding_mask = torch.arange(max_width, **factory_kwargs) >= width.unsqueeze(dim=-1)

            num_head_tokens = 0

            if self.embedding.insert_cls_token:
                num_head_tokens += 1

            if self.embedding.insert_dist_token:
                num_head_tokens += 1

            padding_mask = F.pad(padding_mask, (num_head_tokens, 0), value=False)

        return padding_mask

    @classmethod
    def build_from_default_config(cls, is_teacher: bool = True) -> "MusicTaggingTransformer":
        # PositionalPatchEmbedding
        hidden_channels = 128
        kernel_size = 3
        pool_kernel_size = None
        num_embedding_layers = 3
        num_embedding_blocks = 2
        insert_cls_token = True
        insert_dist_token = False
        embedding_dropout = 0.1
        max_length = 512
        support_extrapolation = False

        # MusicTaggingTransformerEncoder
        if is_teacher:
            d_model = 256
        else:
            d_model = 64

        dim_feedforward = 4 * d_model
        nhead = 8
        activation = "gelu"
        backbone_dropout = 0.1
        num_backbone_layers = 4
        layer_norm_eps = 1e-5
        batch_first = True
        norm_first = True
        bias = True
        norm = None

        n_bins = 128
        num_classes = 50

        embedding = PositionalPatchEmbedding(
            d_model,
            hidden_channels,
            n_bins,
            kernel_size=kernel_size,
            pool_kernel_size=pool_kernel_size,
            num_layers=num_embedding_layers,
            num_blocks=num_embedding_blocks,
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
            dropout=embedding_dropout,
            max_length=max_length,
            support_extrapolation=support_extrapolation,
        )
        backbone = MusicTaggingTransformerEncoder(
            d_model,
            nhead,
            num_layers=num_backbone_layers,
            dim_feedforward=dim_feedforward,
            dropout=backbone_dropout,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            bias=bias,
            norm=norm,
        )
        aggregator = HeadTokensAggregator(
            insert_cls_token=insert_cls_token,
            insert_dist_token=insert_dist_token,
        )
        head = MLPHead(d_model, num_classes)

        model = cls(embedding, backbone, aggregator=aggregator, head=head)

        return model

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "MusicTaggingTransformer":
        """Build pretrained Music Tagging Transformer.

        The weights are extracted from official implementation.

        Examples:

            >>> import torch
            >>> import torch.nn.functional as F
            >>> from audyn.transforms import MusicTaggingTransformerMelSpectrogram
            >>> from audyn.models import MusicTaggingTransformer
            >>> from audyn.utils.data.msd import tags
            >>> torch.manual_seed(0)
            >>> transform = MusicTaggingTransformerMelSpectrogram.build_from_pretrained()
            >>> model = MusicTaggingTransformer.build_from_pretrained("music-tagging-transformer_teacher")
            >>> waveform = torch.randn((4, 30 * transform.sample_rate))
            >>> spectrogram = transform(waveform)
            >>> logit = model(spectrogram)
            >>> likelihood = F.sigmoid(logit)
            >>> print(likelihood.size())
            torch.Size([4, 50])
            >>> print(len(tags))
            50  # 50 classes in MSD dataset

        .. note::

            Supported pretrained model names are
                - music-tagging-transformer_teacher
                - music-tagging-transformer_student

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

            target = pretrained_model_config["_target_"]
            old_class_names = [
                "audyn.models.music_tagging_transformer.MusicTaggingTransformer",
                "audyn.models.MusicTaggingTransformer",
            ]
            new_class_name = f"{cls.__module__}.{cls.__name__}"

            for old_class_name in old_class_names:
                target = target.replace(old_class_name, new_class_name)

            pretrained_model_config["_target_"] = target
            model: MusicTaggingTransformer = instantiate(pretrained_model_config)
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


class MusicTaggingTransformerLinearProbing(MusicTaggingTransformer):
    """Music Tagging Transformer [#won2021semi]_ for linear probing.

    Unlike ``MusicTaggingTransformer``, gradients of ``embedding``, ``backbone``,
    and ``aggregator`` are removed.
    """

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: MusicTaggingTransformerEncoder,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(
            embedding=embedding,
            backbone=backbone,
            aggregator=aggregator,
            head=head,
        )

        for p in self.embedding.parameters():
            p.requires_grad = False
            p.grad = None

        for p in self.backbone.parameters():
            p.requires_grad = False
            p.grad = None

        for p in self.aggregator.parameters():
            p.requires_grad = False
            p.grad = None


def _create_pretrained_model_configs() -> Dict[str, Dict[str, str]]:
    """Create pretrained_model_configs without circular import error."""

    from ..utils import model_cache_dir

    pretrained_model_configs = {
        "music-tagging-transformer_teacher": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.2/music-tagging-transformer_teacher.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "MusicTaggingTransformer",
                "music-tagging-transformer_teacher.pth",
            ),
        },
        "music-tagging-transformer_student": {
            "url": "https://github.com/tky823/Audyn/releases/download/v0.0.2/music-tagging-transformer_student.pth",  # noqa: E501
            "path": os.path.join(
                model_cache_dir,
                "MusicTaggingTransformer",
                "music-tagging-transformer_student.pth",
            ),
        },
    }

    return pretrained_model_configs
