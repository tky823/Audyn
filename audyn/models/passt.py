import copy
import os
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.common_types import _size_2_t

from ..modules.passt import DisentangledPositionalPatchEmbedding, Patchout
from ..utils.github import download_file_from_github_release
from .ast import Aggregator, BaseAudioSpectrogramTransformer, Head

__all__ = [
    "PaSST",
]


class PaSST(BaseAudioSpectrogramTransformer):
    """Patchout faSt Spectrogram Transformer (PaSST).

    Args:
        embedding (audyn.modules.passt.DisentangledPositionalPatchEmbedding): Patch embedding
            followed by positional embeddings disentangled by frequency and time ones.
        dropout (audyn.models.passt.PatchDropout): Patch dropout module. The expected input
            is 4D feature (batch_size, embedding_dim, height, width). The expected output is
            tuple of 3D feature (batch_size, max_length, embedding_dim) and length (batch_size,).
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: DisentangledPositionalPatchEmbedding,
        dropout: Patchout,
        backbone: nn.TransformerEncoder,
        aggregator: Optional[Aggregator] = None,
        head: Optional[Head] = None,
    ) -> None:
        super(BaseAudioSpectrogramTransformer, self).__init__()

        self.embedding = embedding
        self.dropout = dropout
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        stride: Optional[_size_2_t] = None,
        n_bins: Optional[int] = None,
        n_frames: Optional[int] = None,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "PaSST":
        """Build pretrained PaSST.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from audyn.models.passt import PaSST
            >>> model = PaSST.build_from_pretrained("passt-base-stride10-struct-ap0.476-swa")

        .. note::

            Supported pretrained model names are
                - passt-base-stride10-struct-ap0.476-swa

        """  # noqa: E501
        from ..utils import instantiate, model_cache_dir

        pretrained_model_configs = {
            "passt-base-stride10-struct-ap0.476-swa": {
                "url": "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/passt-base-stride10-struct-ap0.476-swa.pth",  # noqa: E501
                "path": os.path.join(
                    model_cache_dir,
                    "PaSST",
                    "passt-base-stride10-struct-ap0.476-swa.pth",
                ),
            },
        }
        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path, map_location=lambda storage, loc: storage
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            model: PaSST = instantiate(pretrained_model_config)
            model.load_state_dict(model_state_dict)

            if aggregator is not None:
                model.aggregator = aggregator

            if head is not None:
                model.head = head

            # update patch embedding if necessary
            model.embedding = _align_patch_embedding(
                model.embedding, stride=stride, n_bins=n_bins, n_frames=n_frames
            )

            return model
        elif pretrained_model_name_or_path in pretrained_model_configs:
            config = pretrained_model_configs[pretrained_model_name_or_path]
            url = config["url"]
            path = config["path"]
            download_file_from_github_release(url, path=path)
            model = cls.build_from_pretrained(
                path,
                stride=stride,
                n_bins=n_bins,
                n_frames=n_frames,
                aggregator=aggregator,
                head=head,
            )

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PaSST.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Output features. The shape is one of

                - (batch_size, out_channels) if ``aggregator`` and ``head`` are given.
                - (batch_size, max_length, embedding_dim) if neither of ``aggregator`` nor ``head`
                    is given.
                - (batch_size, embedding_dim) if only ``aggregator`` is given.

        """
        x = self.embedding(input)
        x_patch = self.spectrogram_to_patches(input)

        # NOTE: During training, ``embedding`` uses random operation,
        #       but ``spectrogram_to_patches`` does not,
        #       which means each patch in ``x_patch`` does not correspond to
        #       the one in the same position in ``x``.
        _, _, height, width = x_patch.size()

        head_tokens, x = self.split_sequence(x)
        x = self.sequence_to_patches(x, height=height, width=width)
        x, _ = self.dropout(x)

        assert (
            x.dim() == 3
        ), "Return of dropout should be 3D (batch_size, max_length, embedding_dim)."

        x = self.prepend_tokens(x, tokens=head_tokens)
        output = self.transformer_forward(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output


def _align_patch_embedding(
    orig_patch_embedding: DisentangledPositionalPatchEmbedding,
    stride: Optional[_size_2_t] = None,
    n_bins: Optional[int] = None,
    n_frames: Optional[int] = None,
    support_extrapolation: bool = False,
) -> DisentangledPositionalPatchEmbedding:
    pretrained_embedding_dim = orig_patch_embedding.embedding_dim
    pretrained_kernel_size = orig_patch_embedding.kernel_size
    pretrained_stride = orig_patch_embedding.stride
    pretrained_insert_cls_token = orig_patch_embedding.insert_cls_token
    pretrained_insert_dist_token = orig_patch_embedding.insert_dist_token
    pretrained_n_bins = orig_patch_embedding.n_bins
    pretrained_n_frames = orig_patch_embedding.n_frames
    pretrained_conv2d = orig_patch_embedding.conv2d
    pretrained_frequency_embedding = orig_patch_embedding.frequency_embedding
    pretrained_time_embedding = orig_patch_embedding.time_embedding
    pretrained_cls_token = orig_patch_embedding.cls_token
    pretrained_dist_token = orig_patch_embedding.dist_token

    if stride is None:
        stride = pretrained_stride

    if n_bins is None:
        n_bins = pretrained_n_bins

    if n_frames is None:
        n_frames = pretrained_n_frames

    new_patch_embedding = DisentangledPositionalPatchEmbedding(
        pretrained_embedding_dim,
        kernel_size=pretrained_kernel_size,
        stride=stride,
        insert_cls_token=pretrained_insert_cls_token,
        insert_dist_token=pretrained_insert_dist_token,
        n_bins=n_bins,
        n_frames=n_frames,
        support_extrapolation=support_extrapolation,
    )

    conv2d_state_dict = copy.deepcopy(pretrained_conv2d.state_dict())
    new_patch_embedding.conv2d.load_state_dict(conv2d_state_dict)

    # Triming of positional embeddings should be deterministic, so training is set to False.
    pretrained_frequency_embedding = new_patch_embedding.resample_frequency_embedding(
        pretrained_frequency_embedding,
        n_bins,
        training=False,
    )
    pretrained_time_embedding = new_patch_embedding.resample_time_embedding(
        pretrained_time_embedding,
        n_frames,
        training=False,
    )
    new_patch_embedding.frequency_embedding.data.copy_(pretrained_frequency_embedding)
    new_patch_embedding.time_embedding.data.copy_(pretrained_time_embedding)

    if pretrained_insert_cls_token:
        new_patch_embedding.cls_token.data.copy_(pretrained_cls_token)

    if pretrained_insert_dist_token:
        new_patch_embedding.dist_token.data.copy_(pretrained_dist_token)

    return new_patch_embedding
