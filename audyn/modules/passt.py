import warnings
from abc import abstractmethod
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair

__all__ = [
    "DisentangledPositionalPatchEmbedding",
    "Patchout",
    "UnstructuredPatchout",
    "StructuredPatchout",
]


class DisentangledPositionalPatchEmbedding(nn.Module):
    """Patch embedding + trainable frequency and time embeddings.

    Args:
        embedding_dim (int): Embedding dimension.
        kernel_size (_size_2_t): Kernel size that corresponds to patch.
        stride (_size_2_t): Stride.
        insert_cls_token (bool): If ``True``, class token is inserted to beginning of sequence.
        insert_dist_token (bool): If ``True``, distillation token is inserd to beginning sequence.
        dropout (float): Dropout rate.
        n_bins (int): Number of input bins.
        n_frames (int): Number of input frames.
        support_extrapolation (bool): If ``True``, embeddings are extrapolated for longer input
            as in audio spectrogram transformer. Otherwise, input sequence is trimmed.
            Default: ``False``.

    .. note::

        Unlike official implementation, trainable positional embedding for CLS (and DIST) token(s)
        are omitted in terms of redundancy.

    """

    def __init__(
        self,
        embedding_dim: int,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        insert_cls_token: bool = False,
        insert_dist_token: bool = False,
        dropout: float = 0,
        n_bins: int = None,
        n_frames: int = None,
        support_extrapolation: bool = False,
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

        if insert_dist_token and not insert_cls_token:
            raise ValueError("When insert_dist_token=True, insert_cls_token should be True.")

        kernel_size = _pair(kernel_size)

        if stride is None:
            stride = kernel_size

        stride = _pair(stride)

        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.insert_cls_token = insert_cls_token
        self.insert_dist_token = insert_dist_token
        self.n_bins = n_bins
        self.n_frames = n_frames
        self.support_extrapolation = support_extrapolation

        self.conv2d = nn.Conv2d(
            1,
            embedding_dim,
            kernel_size=kernel_size,
            stride=stride,
        )

        height, width = self.compute_output_shape(n_bins, n_frames)
        frequency_embedding = torch.empty((embedding_dim, height), **factory_kwargs)
        time_embedding = torch.empty((embedding_dim, width), **factory_kwargs)
        self.frequency_embedding = nn.Parameter(frequency_embedding)
        self.time_embedding = nn.Parameter(time_embedding)

        num_head_tokens = 0

        if insert_cls_token:
            num_head_tokens += 1
            cls_token = torch.empty(
                (embedding_dim,),
                **factory_kwargs,
            )
            self.cls_token = nn.Parameter(cls_token)
        else:
            self.register_parameter("cls_token", None)

        if insert_dist_token:
            num_head_tokens += 1
            dist_token = torch.empty(
                (embedding_dim,),
                **factory_kwargs,
            )
            self.dist_token = nn.Parameter(dist_token)
        else:
            self.register_parameter("dist_token", None)

        self.dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # based on official implementation
        nn.init.trunc_normal_(self.frequency_embedding.data, std=0.02)
        nn.init.trunc_normal_(self.time_embedding.data, std=0.02)

        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token.data, std=0.02)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token.data, std=0.02)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of FactorizedPositionalPatchEmbedding.

        Args:
            input (torch.Tensor): Spectrogram of shape (batch_size, n_bins, n_frames).

        Returns:
            torch.Tensor: (batch_size, height * width + num_head_tokens, embedding_dim),
                where `num_head_tokens` represents number of tokens for [CLS] and [DIST].

        """
        frequency_embedding = self.frequency_embedding
        time_embedding = self.time_embedding

        batch_size, n_bins, n_frames = input.size()
        height, width = self.compute_output_shape(n_bins, n_frames)
        height_org = frequency_embedding.size(-1)
        width_org = time_embedding.size(-1)
        x = input.unsqueeze(dim=-3)
        x = self.conv2d(x)

        # resample embeddings
        if height > height_org and not self.support_extrapolation:
            warnings.warn(
                "Number of frequency bins is greater than predefined value.",
                UserWarning,
                stacklevel=2,
            )
            x, _ = torch.split(x, [height_org, height - height_org], dim=-2)
        else:
            frequency_embedding = self.resample_frequency_embedding(
                frequency_embedding,
                n_bins,
                training=self.training,
            )

        if width > width_org and not self.support_extrapolation:
            warnings.warn(
                "Number of time frames is greater than predefined value.",
                UserWarning,
                stacklevel=2,
            )
            x, _ = torch.split(x, [width_org, width - width_org], dim=-1)
        else:
            time_embedding = self.resample_time_embedding(
                time_embedding,
                n_frames,
                training=self.training,
            )

        x = x + frequency_embedding.unsqueeze(dim=-1) + time_embedding.unsqueeze(dim=-2)

        x = self.patches_to_sequence(x)

        if self.insert_dist_token:
            dist_token = self.dist_token.expand((batch_size, 1, -1))
            x = torch.cat([dist_token, x], dim=-2)

        if self.insert_cls_token:
            cls_token = self.cls_token.expand((batch_size, 1, -1))
            x = torch.cat([cls_token, x], dim=-2)

        output = self.dropout(x)

        return output

    def spectrogram_to_patches(self, input: torch.Tensor) -> torch.Tensor:
        """Convert spectrogram to patches."""
        conv2d = self.conv2d
        frequency_embedding = self.frequency_embedding
        time_embedding = self.time_embedding

        batch_size, n_bins, n_frames = input.size()
        height_org = frequency_embedding.size(-1)
        width_org = time_embedding.size(-1)
        x = input.view(batch_size, 1, n_bins, n_frames)
        x = F.unfold(
            x,
            kernel_size=conv2d.kernel_size,
            dilation=conv2d.dilation,
            padding=conv2d.padding,
            stride=conv2d.stride,
        )
        height, width = self.compute_output_shape(n_bins, n_frames)
        x = x.view(batch_size, -1, height, width)

        # resample embeddings
        if height > height_org and not self.support_extrapolation:
            x, _ = torch.split(x, [height_org, height - height_org], dim=-2)

        if width > width_org and not self.support_extrapolation:
            x, _ = torch.split(x, [width_org, width - width_org], dim=-1)

        output = x

        return output

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

    def split_sequence(self, sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split sequence to head tokens and content tokens.

        Args:
            sequence (torch.Tensor): Sequence containing head tokens, i.e. class and distillation
                tokens. The shape is (batch_size, length, embedding_dim).

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Head tokens of shape (batch_size, num_head_tokens, embedding_dim).
                - torch.Tensor: Sequence of shape
                    (batch_size, length - num_head_tokens, embedding_dim).

        .. note::

            This method is applicable even when sequence does not contain head tokens. In that
            case, an empty sequnce is returened as the first item of returned tensors.

        """
        length = sequence.size(-2)
        num_head_tokens = 0

        if self.cls_token is not None:
            num_head_tokens += 1

        if self.dist_token is not None:
            num_head_tokens += 1

        head_tokens, sequence = torch.split(
            sequence, [num_head_tokens, length - num_head_tokens], dim=-2
        )

        return head_tokens, sequence

    def resample_frequency_embedding(
        self,
        embedding: Union[torch.Tensor],
        n_bins: int,
        mode: str = "bilinear",
        training: Optional[bool] = None,
    ) -> torch.Tensor:
        _, height_org = embedding.size()
        height, _ = self.compute_output_shape(n_bins, self.n_frames)

        if height_org > height:
            if training is None:
                training = self.training

            if training:
                start_idx = torch.randint(0, height_org - height, ()).item()
            else:
                start_idx = 0

            _, embedding, _ = torch.split(
                embedding,
                [start_idx, height, height_org - height - start_idx],
                dim=-1,
            )
        elif height > height_org:
            embedding = embedding.view(1, -1, height_org, 1)
            embedding = F.interpolate(embedding, size=(height, 1), mode=mode)
            embedding = embedding.view(-1, height)

        return embedding

    def resample_time_embedding(
        self,
        embedding: Union[torch.Tensor],
        n_frames: int,
        mode: str = "bilinear",
        training: Optional[bool] = None,
    ) -> torch.Tensor:
        _, width_org = embedding.size()
        _, width = self.compute_output_shape(self.n_bins, n_frames)

        if width_org > width:
            if training is None:
                training = self.training

            if training:
                start_idx = torch.randint(0, width_org - width, ()).item()
            else:
                start_idx = 0

            _, embedding, _ = torch.split(
                embedding,
                [start_idx, width, width_org - width - start_idx],
                dim=-1,
            )
        elif width > width_org:
            embedding = embedding.view(1, -1, 1, width_org)
            embedding = F.interpolate(embedding, size=(1, width), mode=mode)
            embedding = embedding.view(-1, width)

        return embedding

    def compute_output_shape(self, n_bins: int, n_frames: int) -> Tuple[int, int]:
        Kh, Kw = self.conv2d.kernel_size
        Sh, Sw = self.conv2d.stride
        height = (n_bins - Kh) // Sh + 1
        width = (n_frames - Kw) // Sw + 1

        return height, width


class Patchout(nn.Module):
    """Base class of Patchout."""

    def __init__(self, sample_wise: bool = False) -> None:
        super().__init__()

        self.sample_wise = sample_wise

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of Patchout.

        Args:
            input (torch.Tensor): Patches of shape (batch_size, embedding_dim, height, width).

        Returns:
            Tuple: Tuple of tensors containing:

                - torch.Tensor: Kept patches of shape (batch_size, max_length, embedding_dim).
                - torch.LongTensor: Length of shape (batch_size,).

        """
        batch_size, _, height, width = input.size()

        if self.training:
            if self.sample_wise:
                keeping_mask = []

                for unbatched_input in input:
                    _keeping_mask = self.create_unbatched_keeping_mask(unbatched_input)
                    keeping_mask.append(_keeping_mask)

                keeping_mask = torch.stack(keeping_mask, dim=0)
            else:
                keeping_mask = self.create_unbatched_keeping_mask(input[0])
                keeping_mask = keeping_mask.expand((batch_size, -1, -1))

            output, length = self.select_kept_patches(input, keeping_mask=keeping_mask)
        else:
            x = input.view(batch_size, -1, height * width)
            output = x.permute(0, 2, 1).contiguous()
            length = torch.full(
                (batch_size,),
                fill_value=height * width,
                dtype=torch.bool,
                device=output.device,
            )

        return output, length

    @abstractmethod
    def create_unbatched_keeping_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        """Create keeping mask of shape (height, width).

        Args:
            input (torch.Tensor): Unbatched patches of shape (embedding_dim, height, width).

        Returns:
            torch.BoolTensor: Mask of shape (height, width).

        """
        pass

    def select_kept_patches(
        self, input: torch.Tensor, keeping_mask: torch.BoolTensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Select kept patches.

        Args:
            input (torch.Tensor): Estimated sequence of shape
                (batch_size, embedding_dim, height, width).
            keeping_mask (torch.BoolTensor): Keeping mask of shape (batch_size, height, width).
                ``True`` is treated as position to keep.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Selected sequence of shape (batch_size, max_length, embedding_dim).
                - torch.LongTensor: Length of shape (batch_size,).

        """
        batch_size, embedding_dim, height, width = input.size()

        assert keeping_mask.size() == (batch_size, height, width)

        x = input.view(batch_size, embedding_dim, height * width)
        keeping_mask = keeping_mask.view(batch_size, height * width)
        output = []

        for _x, _mask in zip(x, keeping_mask):
            _x = _x.masked_select(_mask)
            _output = _x.view(embedding_dim, -1)
            _output = _output.permute(1, 0).contiguous()
            output.append(_output)

        output = nn.utils.rnn.pad_sequence(output, batch_first=True)
        keeping_mask = keeping_mask.to(torch.long)
        length = keeping_mask.sum(dim=-1)

        return output, length


class UnstructuredPatchout(Patchout):
    """Unstructured patch out, which drops patches at random.

    Args:
        num_drops (int): Number of patches to drop out.
        sample_wise (bool): If ``True``, patch-out is applied per sample.

    """

    def __init__(
        self,
        num_drops: int,
        sample_wise: bool = False,
    ) -> None:
        super().__init__(sample_wise=sample_wise)

        self.num_drops = num_drops

    def create_unbatched_keeping_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        """Create keeping mask of shape (height, width).

        Args:
            input (torch.Tensor): Unbatched patches of shape (embedding_dim, height, width).

        Returns:
            torch.BoolTensor: Mask of shape (height, width).

        """
        num_drops = self.num_drops

        _, height, width = input.size()

        indices = torch.randperm(height * width)[:num_drops]
        padding_mask = torch.zeros(
            (height * width,),
            dtype=torch.bool,
            device=input.device,
        )
        padding_mask.scatter_(0, indices, True)
        padding_mask = padding_mask.view(height, width)
        keeping_mask = torch.logical_not(padding_mask)

        return keeping_mask


class StructuredPatchout(Patchout):
    """Structured patch out, which drops vertical or horizontal patches at random.

    Args:
        num_frequency_drops (int): Number of frequency bins to drop out.
        num_time_drops (int): Number of time frames to drop out.
        sample_wise (bool): If ``True``, masking is applied per sample.

    """

    def __init__(
        self,
        num_frequency_drops: int = 0,
        num_time_drops: int = 0,
        sample_wise: bool = False,
    ) -> None:
        super().__init__(sample_wise=sample_wise)

        self.num_frequency_drops = num_frequency_drops
        self.num_time_drops = num_time_drops
        self.sample_wise = sample_wise

    def create_unbatched_keeping_mask(self, input: torch.Tensor) -> torch.BoolTensor:
        """Create keeping mask of shape (height, width).

        Args:
            input (torch.Tensor): Unbatched patches of shape (embedding_dim, height, width).

        Returns:
            torch.BoolTensor: Mask of shape (height, width).

        """
        num_frequency_drops = self.num_frequency_drops
        num_time_drops = self.num_time_drops

        _, height, width = input.size()

        # frequency mask
        indices = torch.randperm(height)[:num_frequency_drops]
        frequency_padding_mask = torch.zeros(
            (height,),
            dtype=torch.bool,
            device=input.device,
        )
        frequency_padding_mask.scatter_(0, indices, True)

        # time mask
        indices = torch.randperm(width)[:num_time_drops]
        time_padding_mask = torch.zeros(
            (width,),
            dtype=torch.bool,
            device=input.device,
        )
        time_padding_mask.scatter_(0, indices, True)

        padding_mask = torch.logical_or(
            frequency_padding_mask.unsqueeze(dim=-1), time_padding_mask.unsqueeze(dim=-2)
        )
        keeping_mask = torch.logical_not(padding_mask)

        return keeping_mask
