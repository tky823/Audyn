"""Self-supervised audio spectorgram transformer."""

import math
import warnings
from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = [
    "SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
    "MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel",
    "PositionalPatchEmbedding",
    "Masker",
    "MLP",
    "Aggregator",
    "AverageAggregator",
    "Head",
    "MLPHead",
    "SSASTMPM",
    "MultiTaskSSASTMPM",
]


class SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel(nn.Module):
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
        embedding: "PositionalPatchEmbedding",
        masker: "Masker",
        backbone: nn.TransformerEncoder,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.masker = masker
        self.backbone = backbone

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError("Forward pass is not implemented.")

    def patch_transformer_forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Transformer with patch inputs.

        Args:
            input (torch.Tensor): Patch feature of shape
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        _, _, height, width = input.size()

        x = self.patches_to_sequence(input)
        x = self.backbone(x)
        output = self.sequence_to_patches(x, height=height, width=width)

        return output

    @torch.no_grad()
    def inference(self, input: torch.Tensor) -> torch.Tensor:
        """Inference by Transformer backbone.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        x = self.embedding(input)
        output = self.patch_transformer_forward(x)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches.

        Actual implementation depends on ``self.embedding.spectrogram_to_patches``.

        """
        return self.embedding.spectrogram_to_patches(input)

    def patches_to_sequence(self, input: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
        """Convert 3D (batch_size, height, width) or 4D (batch_size, embedding_dim, height, width)
        tensor to shape (batch_size, length, *) for input of Transformer.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, height, width) or
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Sequence of shape (batch_size, length) or
                (batch_size, length, embedding_dim).

        """
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, height, width = input.size()
            output = input.view(batch_size, height * width)
        elif n_dims == 4:
            batch_size, embedding_dim, height, width = input.size()
            x = input.view(batch_size, embedding_dim, height * width)
            output = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError("Only 3D and 4D tensors are supported.")

        return output

    def sequence_to_patches(
        self, input: Union[torch.Tensor, torch.BoolTensor], height: int, width: int
    ) -> torch.Tensor:
        """Convert (batch_size, max_length, *) tensor to 3D (batch_size, height, width)
        or 4D (batch_size, embedding_dim, height, width) one.
        This method corresponds to inversion of ``patches_to_sequence``.
        """
        n_dims = input.dim()

        if n_dims == 2:
            batch_size, _ = input.size()
            output = input.view(batch_size, height, width)
        elif n_dims == 3:
            batch_size, _, embedding_dim = input.size()
            x = input.view(batch_size, height, width, embedding_dim)
            output = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Only 2D and 3D tensors are supported.")

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


class SelfSupervisedAudioSpectrogramTransformer(nn.Module):
    """Self-supervised audio spectrogram transformer.

    Args:
        embedding (audyn.models.ssast.PositionalPatchEmbedding): Patch embedding
            followed by positional embedding.
        masker (audyn.models.ssast.Masker): Masking module that replaces some patches
            with mask tokens.
        backbone (nn.TransformerEncoder): Transformer (encoder).

    """

    def __init__(
        self,
        embedding: "PositionalPatchEmbedding",
        backbone: nn.TransformerEncoder,
        aggregator: Optional["Aggregator"] = None,
        head: Optional["Head"] = None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.aggregator = aggregator
        self.head = head

        if self.aggregator is None and self.head is not None:
            warnings.warn(
                "Head is given, but aggregator is not given, "
                "which may lead to unexpected behavior."
            )

    def forward(self, input: torch.Tensor) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.LongTensor],
        Tuple[torch.Tensor, torch.Tensor, torch.LongTensor],
    ]:
        """Forward pass of SelfSupervisedAudioSpectrogramTransformer.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        x = self.embedding(input)
        output = self.patch_transformer_forward(x)

        if self.aggregator is not None:
            output = self.aggregator(output)

        if self.head is not None:
            output = self.head(output)

        return output

    def patch_transformer_forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Transformer with patch inputs.

        Args:
            input (torch.Tensor): Patch feature of shape
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Estimated patches of shape (batch_size, embedding_dim, height, width).

        """
        _, _, height, width = input.size()

        x = self.patches_to_sequence(input)
        x = self.backbone(x)
        output = self.sequence_to_patches(x, height=height, width=width)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches.

        Actual implementation depends on ``self.embedding.spectrogram_to_patches``.

        """
        return self.embedding.spectrogram_to_patches(input)

    def patches_to_sequence(self, input: Union[torch.Tensor, torch.BoolTensor]) -> torch.Tensor:
        """Convert 3D (batch_size, height, width) or 4D (batch_size, embedding_dim, height, width)
        tensor to shape (batch_size, length, *) for input of Transformer.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, height, width) or
                (batch_size, embedding_dim, height, width).

        Returns:
            torch.Tensor: Sequence of shape (batch_size, length) or
                (batch_size, length, embedding_dim).

        """
        n_dims = input.dim()

        if n_dims == 3:
            batch_size, height, width = input.size()
            output = input.view(batch_size, height * width)
        elif n_dims == 4:
            batch_size, embedding_dim, height, width = input.size()
            x = input.view(batch_size, embedding_dim, height * width)
            output = x.permute(0, 2, 1).contiguous()
        else:
            raise ValueError("Only 3D and 4D tensors are supported.")

        return output

    def sequence_to_patches(
        self, input: Union[torch.Tensor, torch.BoolTensor], height: int, width: int
    ) -> torch.Tensor:
        """Convert (batch_size, max_length, *) tensor to 3D (batch_size, height, width)
        or 4D (batch_size, embedding_dim, height, width) one.
        This method corresponds to inversion of ``patches_to_sequence``.
        """
        n_dims = input.dim()

        if n_dims == 2:
            batch_size, _ = input.size()
            output = input.view(batch_size, height, width)
        elif n_dims == 3:
            batch_size, _, embedding_dim = input.size()
            x = input.view(batch_size, height, width, embedding_dim)
            output = x.permute(0, 3, 1, 2).contiguous()
        else:
            raise ValueError("Only 2D and 3D tensors are supported.")

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

        # for reconstruction
        x, masking_mask = self.masker(x)
        x = self.patch_transformer_forward(x)
        reconstruction_output, reconstruction_length = self.select_masked_patches(
            x, masking_mask=masking_mask
        )
        reconstruction_target, _ = self.select_masked_patches(
            target,
            masking_mask=masking_mask,
        )
        reconstruction_output = self.reconstructor(reconstruction_output)

        # for classification
        x, masking_mask = self.masker(x)
        x = self.patch_transformer_forward(x)
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


class PositionalPatchEmbedding(nn.Module):
    """Patch embedding + trainable positional embedding.

    Args:
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
        n_bins (int): Number of input bins.
        n_frames (int): Number of input frames.

    """

    def __init__(
        self,
        embedding_dim: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        n_bins: int = None,
        n_frames: int = None,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        super().__init__()

        if n_bins is None:
            raise ValueError("n_bins is required.")

        if n_frames is None:
            raise ValueError("n_frames is required.")

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.conv2d = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        height, width = self.compute_output_shape(n_bins, n_frames)
        positional_embedding = torch.empty((embedding_dim, height, width), **factory_kwargs)
        self.positional_embedding = nn.Parameter(positional_embedding)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # based on official implementation
        nn.init.trunc_normal_(self.positional_embedding.data, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of PositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, embedding_dim, height, width).

        """
        positional_embedding = self.positional_embedding
        _, n_bins, n_frames = input.size()
        x = input.unsqueeze(dim=-3)
        x = self.conv2d(x)
        output = x + self.resample_positional_embedding(
            positional_embedding,
            n_bins,
            n_frames,
        )

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches."""
        conv2d = self.conv2d
        batch_size, n_bins, n_frames = input.size()
        x = input.view(batch_size, 1, n_bins, n_frames)
        x = F.unfold(
            x,
            kernel_size=conv2d.kernel_size,
            dilation=conv2d.dilation,
            padding=conv2d.padding,
            stride=conv2d.stride,
        )
        height, width = self.compute_output_shape(n_bins, n_frames)

        output = x.view(batch_size, -1, height, width)

        return output

    def resample_positional_embedding(
        self, positional_embedding: Union[torch.Tensor], n_bins: int, n_frames: int
    ) -> torch.Tensor:
        """Resample positional embedding.

        Args:
            positional_embedding (torch.Tensor): Positional embedding of shape
                (embedding_dim, height, width).
            n_bins (int): Number of bins, not height.
            n_frames (int): Number of frames, not width.

        Returns:
            torch.Tensor: Resampled positional embedding of shape (embedding_dim, height', width').

        """
        _, height_org, width_org = positional_embedding.size()
        height, width = self.compute_output_shape(n_bins, n_frames)

        if width_org > width:
            start_idx = (width_org - width) // 2
            _, positional_embedding, _ = torch.split(
                positional_embedding,
                [start_idx, width, width_org - width - start_idx],
                dim=-1,
            )
        elif width > width_org:
            positional_embedding = F.interpolate(positional_embedding, size=(width,))

        positional_embedding = positional_embedding.permute(0, 2, 1)

        if height_org > height:
            start_idx = (height_org - height) // 2
            _, positional_embedding, _ = torch.split(
                positional_embedding,
                [start_idx, height, height_org - height - start_idx],
                dim=-1,
            )
        elif height > height_org:
            positional_embedding = F.interpolate(positional_embedding, size=(height,))

        positional_embedding = positional_embedding.permute(0, 2, 1)
        output = positional_embedding.contiguous()

        return output

    def compute_output_shape(self, n_bins: int, n_frames: int) -> Tuple[int, int]:
        Kh, Kw = self.conv2d.kernel_size
        Sh, Sw = self.conv2d.stride
        height = (n_bins - Kh) // Sh + 1
        width = (n_frames - Kw) // Sw + 1

        return height, width


class Masker(nn.Module):
    """Replace some patches with mask token.

    Args:
        embedding_dim (int): Embedding dimension.
        num_masks (int): Number of mask tokens.
        min_cluster (int): Minimum cluster size. Default: ``3``.
        max_cluster (int, optional): Maximum cluster size. Default: ``min_cluster + 3``.
        trainable (bool): If ``True``, mask token is trainable.

    """

    def __init__(
        self,
        embedding_dim: int,
        num_masks: int,
        min_cluster: int = 3,
        max_cluster: Optional[int] = None,
        trainable: bool = True,
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

        if trainable:
            mask_embedding = torch.empty((embedding_dim,), **factory_kwargs)
            self.register_parameter("mask_embedding", nn.Parameter(mask_embedding))
        else:
            self.register_parameter("mask_embedding", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # based on official implementation
        if self.mask_embedding is not None:
            # NOTE: mask_embedding shares data with self.mask_embedding
            #       by .view operation.
            mask_embedding = self.mask_embedding.data.view(1, 1, -1)
            nn.init.xavier_normal_(mask_embedding)

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
        num_masks = self.num_masks

        cluster_size = torch.randint(self.min_cluster, self.max_cluster, ()).item()
        batch_size, _, height, width = input.size()
        mask_height = min(height, cluster_size)
        mask_width = min(width, cluster_size)

        # Considering overlap and triming at edge, we use math.ceil here.
        num_selections = math.ceil(num_masks / (mask_height * mask_width))
        masking_mask = []

        for _ in range(batch_size):
            indices = torch.randperm(height * width)
            indices = indices[:num_selections]
            _masking_mask = torch.zeros((height * width), dtype=input.dtype)
            _masking_mask.scatter_(0, indices, 1)
            masking_mask.append(_masking_mask)

        masking_mask = torch.stack(masking_mask, dim=0)
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
        # masking_mask includes positive values greater than 1,
        # so convert it into False or True.
        masking_mask = masking_mask > 0
        # Then, convert it into 0 or 1.
        masking_mask = masking_mask.to(torch.long)
        masking_mask = masking_mask.to(input.device)

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

        x = attn_mask * input
        x_mask = null_attn_mask * mask_embedding.view(-1, 1, 1)
        output = x + x_mask

        masking_mask = masking_mask.squeeze(dim=-3)
        masking_mask = masking_mask.to(torch.bool)

        return output, masking_mask


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


class Aggregator(nn.Module):
    @abstractmethod
    def forward(
        self, input: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        pass


class AverageAggregator(Aggregator):
    def forward(
        self, input: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        if padding_mask is None:
            batch_size, _, height, width = input.size()
            padding_mask = torch.full(
                (batch_size, height, width),
                fill_value=False,
                dtype=torch.bool,
                device=input.device,
            )

        x = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
        non_padding_mask = torch.logical_not(padding_mask)
        non_padding_mask = non_padding_mask.to(torch.long)
        non_padding_mask = non_padding_mask.sum(dim=(-2, -1))
        output = x.sum(dim=(-2, -1)) / non_padding_mask.unsqueeze(dim=-1)

        return output


class Head(nn.Module):
    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        pass


class MLPHead(Head):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MLPHead.

        Args:
            input (torch.Tensor): Aggregated feature of shape (batch_size, in_channels).

        Returns:
            torch.Tensor: Transformed feature of shape (batch_size, out_channels).

        """
        x = self.norm(input)
        output = self.linear(x)

        return output


class SSASTMPM(SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel):
    """Alias of SelfSupervisedAudioSpectrogramTransformerMaskedPatchModel."""


class MultiTaskSSASTMPM(MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel):
    """Alias of MultiTaskSelfSupervisedAudioSpectrogramTransformerMaskedPatchModel."""
