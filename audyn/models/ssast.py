"""Self-supervised audio spectorgram transformer."""

import math
import os
import warnings
from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.common_types import _size_2_t

from ..modules.vit import PositionalPatchEmbedding
from ..utils import instantiate, model_cache_dir
from ..utils.github import download_file_from_github_release
from .ast import Aggregator, BaseAudioSpectrogramTransformer, Head, MLPHead, _align_patch_embedding

__all__ = [
    "SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
    "MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
    "PositionalPatchEmbedding",  # for backward compatibility
    "Masker",
    "FastMasker",
    "MLP",
    "Head",  # for backward compatibility
    "MLPHead",  # for backward compatibility
    "SSASTMPM",
    "MultiTaskSSASTMPM",
    "SSAST",
]

pretrained_model_configs = {
    "multitask-ssast-patch-base-400": {
        "url": "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/multitask-ssast-patch-base-400.pth",  # noqa: E501
        "path": os.path.join(
            model_cache_dir,
            "SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
            "multitask-ssast-patch-base-400.pth",
        ),
    },
    "multitask-ssast-frame-base-400": {
        "url": "https://github.com/tky823/Audyn/releases/download/v0.0.1.dev3/multitask-ssast-frame-base-400.pth",  # noqa: E501
        "path": os.path.join(
            model_cache_dir,
            "SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
            "multitask-ssast-frame-base-400.pth",
        ),
    },
}


class SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(BaseAudioSpectrogramTransformer):
    """Masked patch model for self-supervised audio spectrogram transformer.

    Args:
        embedding (audyn.models.ssast.PositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        masker (audyn.models.ssast.Masker): Masking module that replaces some patches
            with mask tokens.
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        masker: "Masker",
        backbone: nn.TransformerEncoder,
    ) -> None:
        super(BaseAudioSpectrogramTransformer, self).__init__()

        self.embedding = embedding
        self.masker = masker
        self.backbone = backbone

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Forward pass is not implemented.")

    @torch.no_grad()
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        """Inference by Transformer backbone.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        x = self.embedding(input)

        # just to compute height and width
        target = self.spectrogram_to_patches(input)
        _, _, height, width = target.size()

        x = self.transformer_forward(x)
        _, x = self.split_sequence(x)
        output = self.sequence_to_patches(x, height=height, width=width)

        return output

    def select_masked_patches(
        self, input: torch.Tensor, masking_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Select masked patches.

        Args:
            input (torch.Tensor): Estimated sequence of shape
                (batch_size, embedding_dim, height, width).
            masking_mask (torch.BoolTensor): Masking mask of shape (batch_size, height, width).
                ``True`` is treated as position of mask token.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Selected sequence of shape (batch_size, max_length, embedding_dim).
                - torch.LongTensor: Length of shape (batch_size,).

        """
        batch_size, embedding_dim, height, width = input.size()

        assert masking_mask.size() == (batch_size, height, width)

        x = input.view(batch_size, embedding_dim, height * width)
        masking_mask = masking_mask.view(batch_size, height * width)
        output = []

        for _x, _mask in zip(x, masking_mask):
            _x = _x.masked_select(_mask)
            _output = _x.view(embedding_dim, -1)
            _output = _output.permute(1, 0).contiguous()
            output.append(_output)

        output = nn.utils.rnn.pad_sequence(output, batch_first=True)
        masking_mask = masking_mask.to(torch.long)
        length = masking_mask.sum(dim=-1)

        return output, length

    def prepend_tokens(
        self, sequence: torch.Tensor, tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepaned tokens to sequence.

        Args:
            sequence (torch.Tensor): Sequence of shape (batch_size, length, embedding_dim).
            tokens (torch.Tensor, optional): Tokens of shape
                (batch_size, num_tokens, embedding_dim).

        Returns:
            torch.Tensor: Concatenated sequence of shape
                (batch_size, length + num_tokens, embedding_dim).

        """
        if tokens is None:
            return sequence
        else:
            return torch.cat([tokens, sequence], dim=-2)


class SelfSupervisedAudioSpectrogramTransformer(BaseAudioSpectrogramTransformer):
    """Self-supervised audio spectrogram transformer.

    Args:
        embedding (audyn.models.ssast.PositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: PositionalPatchEmbedding,
        backbone: nn.TransformerEncoder,
        aggregator: Optional["Aggregator"] = None,
        head: Optional["Head"] = None,
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

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        stride: Optional[_size_2_t] = None,
        n_bins: Optional[int] = None,
        n_frames: Optional[int] = None,
        aggregator: Optional[nn.Module] = None,
        head: Optional[nn.Module] = None,
    ) -> "SelfSupervisedAudioSpectrogramTransformer":
        """Build pretrained SelfSupervisedAudioSpectrogramTransformer.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.
            aggregator (nn.Module, optional): Aggregator module.
            head (nn.Module, optional): Head module.

        Examples:

            >>> from audyn.models.ssast import SelfSupervisedAudioSpectrogramTransformer
            >>> model = SelfSupervisedAudioSpectrogramTransformer.build_from_pretrained("multitask-ssast-frame-base-400")

        .. note::

            Supported pretrained model names are
                - multitask-ssast-patch-base-400
                - multitask-ssast-frame-base-400

        """  # noqa: E501
        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path, map_location=lambda storage, loc: storage
            )
            model_state_dict: OrderedDict = state_dict["model"]
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)
            pretrained_model_config = resolved_config.model
            patch_embedding = instantiate(pretrained_model_config.embedding)
            transformer = instantiate(pretrained_model_config.backbone)

            model = cls(
                patch_embedding,
                transformer,
                aggregator=aggregator,
                head=head,
            )

            keys = list(model_state_dict.keys())

            # remove states containing in pretraining model only
            for key in keys:
                if (
                    key.startswith("masker.")
                    or key.startswith("classifier.")
                    or key.startswith("reconstructor.")
                ):
                    _ = model_state_dict.pop(key)

            # add states containing in finetuning model only
            additional_state_dict = OrderedDict()

            for key, value in model.state_dict().items():
                if key.startswith("aggregator.") or key.startswith("head."):
                    additional_state_dict[key] = value

            model_state_dict.update(additional_state_dict)
            model.load_state_dict(model_state_dict)

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
        """Forward pass of SelfSupervisedAudioSpectrogramTransformer.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Estimated patches. The shape is one of
                - (batch_size, height * width + num_head_tokens, embedding_dim).
                - (batch_size, height * width + num_head_tokens, out_channels).
                - (batch_size, embedding_dim).
                - (batch_size, out_channels).

        """
        x = self.embedding(input)
        output = self.transformer_forward(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output


class MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(
    SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel
):
    """Masked patch model for self-supervised audio spectrogram transformer
    that processes reconstruction and classification.

    Args:
        embedding (audyn.models.ssast.PositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        masker (audyn.models.ssast.Masker): Masking module that replaces some patches
            with mask tokens.
        backbone (nn.TransformerEncoder): Transformer (encoder).
        reconstructor (nn.Module): Position-wise reconstructor.
        classifier (nn.Module): Position-wise classifier.

    """

    def __init__(
        self,
        embedding: "PositionalPatchEmbedding",
        masker: "Masker",
        backbone: nn.TransformerEncoder,
        reconstructor: nn.Module,
        classifier: nn.Module,
    ) -> None:
        super().__init__(embedding=embedding, masker=masker, backbone=backbone)

        self.reconstructor = reconstructor
        self.classifier = classifier

    @classmethod
    def build_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        stride: Optional[_size_2_t] = None,
        n_bins: Optional[int] = None,
        n_frames: Optional[int] = None,
        reconstructor: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ) -> "MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel":
        """Build pretrained MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel.

        Args:
            pretrained_model_name_or_path (str): Path to pretrained model or name of pretrained model.

        Examples:

            >>> from audyn.models.ssast import MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel
            >>> model = MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel.build_from_pretrained("multitask-ssast-frame-base-400")

        .. note::

            Supported pretrained model names are
                - multitask-ssast-patch-base-400
                - multitask-ssast-frame-base-400

        """  # noqa: E501
        if os.path.exists(pretrained_model_name_or_path):
            state_dict = torch.load(
                pretrained_model_name_or_path, map_location=lambda storage, loc: storage
            )
            resolved_config = state_dict["resolved_config"]
            resolved_config = OmegaConf.create(resolved_config)

            model: MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel = (
                instantiate(resolved_config.model)
            )
            model.load_state_dict(state_dict["model"])

            # update patch embedding if necessary
            model.embedding = _align_patch_embedding(
                model.embedding, stride=stride, n_bins=n_bins, n_frames=n_frames
            )

            # update reconstructor and classifier if necessary
            if reconstructor is not None:
                model.reconstructor = reconstructor

            if classifier is not None:
                model.classifier = classifier

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
                reconstructor=reconstructor,
                classifier=classifier,
            )

            return model
        else:
            raise FileNotFoundError(f"{pretrained_model_name_or_path} does not exist.")

    def forward(self, input: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.LongTensor],
        Tuple[torch.Tensor, torch.Tensor, torch.LongTensor],
    ]:
        """Forward pass of MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            tuple: Tuple containing

                - tuple: Tuple containing reconstruction output, reconstruction target, and length.
                    Shape of reconstruction output and target is
                    (batch_size, max_length, kernel_height * kernel_width). Shape of length is
                    (batch_size,).
                - tuple: Tuple containing classification output, classification target, and length.
                    Shape of classification output and target is
                    (batch_size, max_length, kernel_height * kernel_width). Shape of length is
                    (batch_size,).

        """
        x = self.embedding(input)
        target = self.spectrogram_to_patches(input)

        _, _, height, width = target.size()

        head_tokens, x = self.split_sequence(x)
        x_patch = self.sequence_to_patches(x, height=height, width=width)

        # for reconstruction
        x, masking_mask = self.masker(x_patch)
        x = self.patches_to_sequence(x)
        x = self.prepend_tokens(x, tokens=head_tokens)
        x = self.transformer_forward(x)
        _, x = self.split_sequence(x)
        x = self.sequence_to_patches(x, height=height, width=width)
        reconstruction_output, reconstruction_length = self.select_masked_patches(
            x, masking_mask=masking_mask
        )
        reconstruction_target, _ = self.select_masked_patches(
            target,
            masking_mask=masking_mask,
        )
        reconstruction_output = self.reconstructor(reconstruction_output)

        # for classification
        x, masking_mask = self.masker(x_patch)
        x = self.patches_to_sequence(x)
        x = self.prepend_tokens(x, tokens=head_tokens)
        x = self.transformer_forward(x)
        _, x = self.split_sequence(x)
        x = self.sequence_to_patches(x, height=height, width=width)
        classification_output, classification_length = self.select_masked_patches(
            x, masking_mask=masking_mask
        )
        classification_target, _ = self.select_masked_patches(
            target,
            masking_mask=masking_mask,
        )
        classification_output = self.classifier(classification_output)

        reconstruction = (reconstruction_output, reconstruction_target, reconstruction_length)
        classification = (classification_output, classification_target, classification_length)

        return reconstruction, classification


class _Masker(nn.Module):
    """Replace some patches with mask token.

    Args:
        embedding_dim (int): Embedding dimension.
        num_masks (int): Number of mask tokens.
        min_cluster (int): Minimum cluster size. Default: ``3``.
        max_cluster (int, optional): Maximum cluster size. Default: ``min_cluster + 3``.
        trainable (bool): If ``True``, mask token is trainable.
        sample_wise (bool): If ``True``, masking is applied per sample.

    """

    def __init__(
        self,
        embedding_dim: int,
        num_masks: int,
        min_cluster: int = 3,
        max_cluster: Optional[int] = None,
        trainable: bool = True,
        sample_wise: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }
        super().__init__()

        if max_cluster is None:
            max_cluster = min_cluster + 3

        self.embedding_dim = embedding_dim
        self.num_masks = num_masks
        self.min_cluster = min_cluster
        self.max_cluster = max_cluster
        self.trainable = trainable
        self.sample_wise = sample_wise

        if trainable:
            mask_embedding = torch.empty((embedding_dim,), **factory_kwargs)
            self.register_parameter("mask_embedding", nn.Parameter(mask_embedding))
        else:
            self.register_parameter("mask_embedding", None)

        self.mask_embedding: Optional[torch.Tensor]

        self._reset_parameters()

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Replace some patches with mask tokens.

        .. note::

            Even when ``self.training = False``, masking is applied to input patches.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, embedding_dim, height, width).

        Returns:
            Tuple: Tuple of tensors containing:

                - torch.Tensor: Masked patches of shape (batch_size, embedding_dim, height, width).
                - torch.BoolTensor: Masking mask of shape (batch_size, height, width).

        """
        masking_mask = self.create_masking_mask(input)

        if self.mask_embedding is None:
            embedding_dim = self.embedding_dim
            factory_kwargs = {
                "device": input.device,
                "dtype": input.dtype,
            }
            mask_embedding = torch.zeros((embedding_dim,), **factory_kwargs)
        else:
            mask_embedding = self.mask_embedding

        null_attn_mask = masking_mask.long()
        attn_mask = 1 - null_attn_mask
        x = attn_mask.unsqueeze(dim=-3) * input
        x_mask = null_attn_mask.unsqueeze(dim=-3) * mask_embedding.view(-1, 1, 1)
        output = x + x_mask

        return output, masking_mask

    def _reset_parameters(self) -> None:
        # based on official implementation
        if self.mask_embedding is not None:
            # NOTE: mask_embedding shares data with self.mask_embedding
            #       by .view operation.
            mask_embedding = self.mask_embedding.data.view(1, 1, -1)
            nn.init.xavier_normal_(mask_embedding)

    @abstractmethod
    def create_masking_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        """Create masking mask from given batched input.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, embedding_dim, height, width).

        Returns:
            torch.BoolTensor: Mask of shape (batch_size, height, width), where `True` represents
                position of mask token in tensor.

        """


class Masker(_Masker):
    def create_masking_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        cluster_size = torch.randint(self.min_cluster, self.max_cluster, ()).item()
        batch_size, _, height, width = input.size()
        mask_height = min(height, cluster_size)
        mask_width = min(width, cluster_size)

        if self.sample_wise:
            masking_mask = []

            for unbatched_input in input:
                _masking_mask = self._create_unbatched_masking_mask(
                    unbatched_input,
                    mask_height=mask_height,
                    mask_width=mask_width,
                    num_masks=self.num_masks,
                )
                masking_mask.append(_masking_mask)

            masking_mask = torch.stack(masking_mask, dim=0)
        else:
            masking_mask = self._create_unbatched_masking_mask(
                input[0],
                mask_height=mask_height,
                mask_width=mask_width,
                num_masks=self.num_masks,
            )
            masking_mask = masking_mask.expand((batch_size, -1, -1))

        masking_mask = masking_mask.to(input.device)

        return masking_mask

    def _create_unbatched_masking_mask(
        self,
        input: torch.Tensor,
        mask_height: int,
        mask_width: int,
        num_masks: Optional[int] = None,
    ) -> torch.BoolTensor:
        """Create masking mask for unbatched input.

        Args:
            input (torch.Tensor): Unbatched feature of shape (embedding_dim, height, width).

        Returns:
            torch.BoolTensor: Unbatched masking mask of shape (height, width).

        """
        if num_masks is None:
            num_masks = self.num_masks

        _, height, width = input.size()
        indices = torch.randperm(height * width).tolist()
        mask_indices = []

        # When number of patches is less than num_masks, set appropriate value.
        num_masks = min(num_masks, len(indices))
        is_finished = False

        while not is_finished:
            idx = indices[0]
            _mask_height = min(mask_height, height - idx // width)
            _mask_width = min(mask_width, width - idx % width)

            for height_offset in range(_mask_height):
                for width_offset in range(_mask_width):
                    offset = width * height_offset + width_offset
                    mask_idx = idx + offset

                    if mask_idx in mask_indices:
                        # duplicated index
                        continue

                    mask_indices.append(mask_idx)
                    indices.remove(mask_idx)

                    if len(mask_indices) >= num_masks:
                        is_finished = True

                    if is_finished:
                        break

                if is_finished:
                    break

        masking_mask = torch.zeros((height * width), dtype=torch.bool)
        mask_indices = torch.tensor(mask_indices)
        masking_mask.scatter_(0, mask_indices, True)
        masking_mask = masking_mask.view(height, width)

        return masking_mask


class FastMasker(_Masker):
    """Replace some patches with mask token.

    Args:
        embedding_dim (int): Embedding dimension.
        num_masks (int): Number of mask tokens.
        min_cluster (int): Minimum cluster size. Default: ``3``.
        max_cluster (int, optional): Maximum cluster size. Default: ``min_cluster + 3``.
        trainable (bool): If ``True``, mask token is trainable.

    .. note::

        The implementation of this class is different from the original implementation
        to increase computing speed.

    """

    def create_masking_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        num_masks = self.num_masks

        cluster_size = torch.randint(self.min_cluster, self.max_cluster, ()).item()
        batch_size, _, height, width = input.size()
        mask_height = min(height, cluster_size)
        mask_width = min(width, cluster_size)

        # Considering overlap and triming at edge, we use math.ceil here.
        num_selections = math.ceil(num_masks / (mask_height * mask_width))

        if self.sample_wise:
            masking_mask = []

            for unbatched_input in input:
                _masking_mask = self._create_unbatched_masking_mask(
                    unbatched_input, num_selections=num_selections
                )
                masking_mask.append(_masking_mask)

            masking_mask = torch.stack(masking_mask, dim=0)
        else:
            masking_mask = self._create_unbatched_masking_mask(
                input[0], num_selections=num_selections
            )
            masking_mask = masking_mask.expand((batch_size, -1))

        masking_mask = masking_mask.unsqueeze(dim=-2)
        masking_mask = masking_mask.expand((-1, mask_height * mask_width, -1))
        masking_mask = F.fold(
            masking_mask,
            output_size=(height + mask_height - 1, width + mask_width - 1),
            kernel_size=(mask_height, mask_width),
        )
        Ph = mask_height - 1
        Pw = mask_width - 1
        padding = (-(Pw // 2), -(Pw - Pw // 2), -(Ph // 2), -(Ph - Ph // 2))
        masking_mask = F.pad(masking_mask, padding)
        masking_mask = masking_mask.squeeze(dim=-3)
        # masking_mask includes positive values greater than 1,
        # so convert it into False or True.
        masking_mask = masking_mask > 0
        masking_mask = masking_mask.to(input.device)

        return masking_mask

    def _create_unbatched_masking_mask(
        self,
        input: torch.Tensor,
        num_selections: int,
    ) -> torch.BoolTensor:
        """Create masking mask for unbatched input.

        Args:
            input (torch.Tensor): Unbatched feature of shape (embedding_dim, height, width).
            num_selections (int): Number of selections for masking. The value may be different
                from ``self.num_masks``.

        Returns:
            torch.BoolTensor: Unbatched masking mask of shape (height * width).

        """
        _, height, width = input.size()

        indices = torch.randperm(height * width)
        indices = indices[:num_selections]
        masking_mask = torch.zeros((height * width), dtype=input.dtype)
        masking_mask.scatter_(0, indices, 1)

        return masking_mask


class MLP(nn.Module):
    """Multi-layer perceptron used for reconstructor and classifier."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels

        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.nonlinear = nn.ReLU()
        self.linear2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.linear1(input)
        x = self.nonlinear(x)
        output = self.linear2(x)

        return output


class SSASTMPM(SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel):
    """Alias of SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel."""


class MultiTaskSSASTMPM(MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel):
    """Alias of MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel."""


class SSAST(SelfSupervisedAudioSpectrogramTransformer):
    """Alias of SelfSupervisedAudioSpectrogramTransformer."""
