from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.modules.positional_encoding import AbsolutePositionalEncoding

available_aggregations = ["cls", "pool", "none"]


class TransformerBackbone(nn.Module):
    cls_embedding: nn.Parameter
    positional_encoding: AbsolutePositionalEncoding
    backbone: nn.TransformerEncoder

    embedding_dim: int
    batch_first: bool
    aggregation: str

    def transformer_forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        batch_first = self.batch_first
        cls_embedding = self.cls_embedding

        factory_kwargs = {"device": input.device}

        if batch_first:
            batch_size, max_length, _ = input.size()
        else:
            max_length, batch_size, _ = input.size()

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)

        if batch_first:
            expanded_padding_mask = padding_mask
        else:
            expanded_padding_mask = padding_mask.permute(1, 0)

        expanded_padding_mask = expanded_padding_mask.unsqueeze(dim=-1)

        x = input.masked_fill(expanded_padding_mask, 0)
        x = self.positional_encoding(x)
        x = x.masked_fill(expanded_padding_mask, 0)
        cls_embedding = cls_embedding.view(1, 1, -1)

        if batch_first:
            cls_embedding = cls_embedding.expand(batch_size, 1, -1)
            x = torch.cat([cls_embedding, x], dim=1)
        else:
            cls_embedding = cls_embedding.expand(1, batch_size, -1)
            x = torch.cat([cls_embedding, x], dim=0)

        padding_mask = F.pad(padding_mask, (1, 0), value=False)

        output = self.backbone(x, src_key_padding_mask=padding_mask)

        return output

    def _reset_parameters(self) -> None:
        self.cls_embedding.data.normal_()


class TextTransformerBackbone(TransformerBackbone):
    """Backbone of text transformer.

    Args:
        vocab_size (int): Vocabulary size excluding mask token.
        embedding_dim (int): Number of embedding channels.
        nhead (int): Number of heads in attention.
        num_layers (int): Number of attention layers.
        batch_first (bool): If ``True``, first dimension is treated as batch.

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            dim_feedforward=embedding_dim,
            batch_first=batch_first,
        )

        self.cls_embedding = nn.Parameter(torch.empty((embedding_dim,)))
        self.word_embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.batch_first = batch_first

        self._reset_parameters()

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        x = self.word_embedding(input)
        output = self.transformer_forward(x, length=length)

        return output


class TextTransformerMaskedLanguageModelBackbone(TextTransformerBackbone):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
        mask_index: int = 0,
        ignore_index: int = -1,
        selection_rate: float = 0.15,
        mask_rate: float = 0.8,
        replace_rate: float = 0.1,
    ) -> None:
        super().__init__(
            vocab_size,
            embedding_dim,
            nhead,
            num_layers=num_layers,
            batch_first=batch_first,
        )

        self.mask_index = mask_index
        self.ignore_index = ignore_index
        self.selection_rate = selection_rate
        self.mask_rate = mask_rate
        self.replace_rate = replace_rate

        assert 0 <= selection_rate <= 1
        assert 0 <= mask_rate <= 1
        assert 0 <= replace_rate <= 1
        assert 0 <= mask_rate + replace_rate <= 1

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of TextTransformerMaskedLanguageModelBackbone.

        Args:
            input (torch.Tensor): Text tokens of shape (batch_size, max_length)
                or (max_length, batch_size).
            length (torch.LongTensor): Lengths of text tokens of shape (batch_size,).

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Logit of shape (batch_size, max_length, embedding_dim)
                    or (max_length, batch_size, embedding_dim).
                - torch.LongTensor: Target tokens of shape (batch_size, max_length)
                    or (max_length, batch_size).

        """
        vocab_size = self.vocab_size
        batch_first = self.batch_first
        mask_index = self.mask_index
        ignore_index = self.ignore_index
        selection_rate = self.selection_rate
        mask_rate = self.mask_rate
        replace_rate = self.replace_rate

        float_factory_kwargs = {
            "dtype": torch.float,
            "device": input.device,
        }
        long_factory_kwargs = {
            "dtype": torch.long,
            "device": input.device,
        }

        if batch_first:
            batch_size, max_length = input.size()
        else:
            max_length, batch_size = input.size()

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **long_factory_kwargs)

        position_indices = torch.arange(max_length, **long_factory_kwargs)

        if batch_first:
            non_padding_mask = position_indices < length.unsqueeze(dim=-1)
        else:
            non_padding_mask = position_indices.unsqueeze(dim=-1) < length

        rand_value = torch.rand(input.size(), **float_factory_kwargs)
        selection_mask = rand_value < selection_rate
        selection_mask = selection_mask & non_padding_mask

        # Mask 1. replace with mask token
        rand_value = torch.rand(input.size(), **float_factory_kwargs)
        masking_mask = selection_mask & (rand_value < mask_rate)

        # Mask 2. replace with different token except for mask token.
        replacement_mask = selection_mask & ((1 - rand_value) < replace_rate)
        replacement_index = torch.randint(0, vocab_size - 1, input.size(), **long_factory_kwargs)
        replacement_index = torch.where(
            replacement_index >= mask_index, replacement_index + 1, replacement_index
        )

        masked_input = input.masked_fill(masking_mask, mask_index)
        masked_input = torch.where(replacement_mask, replacement_index, masked_input)

        # Unselected
        unselection_mask = torch.logical_not(selection_mask)
        target = input.masked_fill(unselection_mask, ignore_index)

        x = self.word_embedding(masked_input)
        output = self.transformer_forward(x, length=length)

        return output, target


class TextTransformerMaskedLanguageModel(nn.Module):
    def __init__(
        self,
        backbone: TextTransformerMaskedLanguageModelBackbone,
        out_proj: nn.Module,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.out_proj = out_proj

        assert self.backbone.batch_first, "Only batch_first=True is supported."

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        x, target = self.backbone(input, length=length)
        max_length = x.size(-2) - 1

        # remove cls token
        _, x = torch.split(x, [1, max_length], dim=-2)
        output = self.out_proj(x)

        return output, target


class TextTransformerTower(nn.Module):
    def __init__(
        self,
        backbone: TextTransformerBackbone,
        aggregator: nn.Module,
    ) -> None:
        super().__init__()

        assert isinstance(backbone, TextTransformerBackbone)
        assert not isinstance(backbone, TextTransformerMaskedLanguageModelBackbone)

        self.backbone = backbone
        self.aggregator = aggregator

    def forward(
        self,
        input: torch.LongTensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        x = self.backbone(input, length=length)
        output = self.aggregator(x)

        return output


class AudioTransformerBackbone(TransformerBackbone):
    """Backbone of audio transformer.

    Args:
        in_channels (int): Number of input channels, which is typically number of frequency bins.
        embedding_dim (int): Number of embedding channels.
        frames_per_patch (int): Number of frames per patch.
        nhead (int): Number of heads in attention.
        num_layers (int): Number of attention layers.
        batch_first (bool): If ``True``, first dimension is treated as batch.
        channels_last (bool): If ``True``, last dimension is treated as channels.

    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        frames_per_patch: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
        channels_last: bool = False,
    ) -> None:
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim,
            nhead,
            dim_feedforward=embedding_dim,
            batch_first=batch_first,
        )

        self.cls_embedding = nn.Parameter(torch.empty((embedding_dim,)))
        self.in_proj = nn.Linear(frames_per_patch * in_channels, embedding_dim)
        self.positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.embedding_dim = embedding_dim
        self.frames_per_patch = frames_per_patch
        self.channels_last = channels_last
        self.batch_first = batch_first

        if (not batch_first) and (not channels_last):
            raise ValueError("Either of batch_first or channels_last should be True.")

        self._reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        frames_per_patch = self.frames_per_patch
        channels_last = self.channels_last
        batch_first = self.batch_first

        factory_kwargs = {
            "device": input.device,
            "dtype": torch.long,
        }

        if batch_first:
            x = input
        else:
            x = input.transpose(1, 0)

        if channels_last:
            x = x.transpose(-2, -1)

        max_length = x.size(-1)
        padding = max_length % frames_per_patch
        batch_size, in_channels, max_length = x.size()
        max_patch_length = (max_length - padding) // frames_per_patch

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        padding_mask = self.generate_sequence_mask(length, max_length - padding)
        patch_padding_mask = self.downsample_padding_mask(padding_mask)
        patch_non_padding_mask = torch.logical_not(patch_padding_mask)
        patch_length = patch_non_padding_mask.sum(dim=-1)

        x = x.unsqueeze(dim=-2)
        x = F.unfold(x, kernel_size=(1, frames_per_patch), stride=(1, frames_per_patch))
        x = x.view(batch_size, in_channels, frames_per_patch, max_patch_length)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size, max_patch_length, frames_per_patch * in_channels)
        x = self.in_proj(x)
        output = self.transformer_forward(x, length=patch_length)

        return output

    def downsample_padding_mask(self, padding_mask: torch.BoolTensor) -> torch.BoolTensor:
        frames_per_patch = self.frames_per_patch

        batch_size, max_length = padding_mask.size()
        max_patch_length = max_length // frames_per_patch
        padding_mask = padding_mask.view(batch_size, max_patch_length, frames_per_patch)
        padding_mask = padding_mask.to(torch.long).sum(dim=-1)
        padding_mask = padding_mask.to(torch.bool)

        return padding_mask

    @staticmethod
    def generate_sequence_mask(length: torch.LongTensor, max_length: int) -> torch.BoolTensor:
        factory_kwargs = {
            "dtype": torch.long,
            "device": length.device,
        }
        padding_mask = torch.arange(max_length, **factory_kwargs) >= length.unsqueeze(dim=-1)

        return padding_mask


class AudioTransformerMaskedPatchModelBackbone(AudioTransformerBackbone):
    def __init__(
        self,
        in_channels: int,
        embedding_dim: int,
        frames_per_patch: int,
        nhead: int,
        num_layers: int = 6,
        batch_first: bool = True,
        channels_last: bool = False,
        selection_rate: float = 0.15,
        cluster_size: int = 3,
    ) -> None:
        super().__init__(
            in_channels,
            embedding_dim,
            frames_per_patch,
            nhead,
            num_layers=num_layers,
            batch_first=batch_first,
            channels_last=channels_last,
        )

        self.selection_rate = selection_rate
        self.cluster_size = cluster_size

        assert self.batch_first, "Only batch_first=True is supported."

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.LongTensor]:
        embedding_dim = self.embedding_dim
        frames_per_patch = self.frames_per_patch
        channels_last = self.channels_last
        batch_first = self.batch_first
        selection_rate = self.selection_rate
        cluster_size = self.cluster_size

        if batch_first:
            x = input
        else:
            x = input.transpose(1, 0)

        if channels_last:
            x = x.transpose(-2, -1)

        max_length = x.size(-1)
        padding = max_length % frames_per_patch
        batch_size, in_channels, max_length = x.size()
        max_patch_length = (max_length - padding) // frames_per_patch

        x = x.unsqueeze(dim=-2)
        x = F.unfold(x, kernel_size=(1, frames_per_patch), stride=(1, frames_per_patch))
        x = x.view(batch_size, in_channels, frames_per_patch, max_patch_length)
        x = x.permute(0, 3, 2, 1).contiguous()
        target = x.view(batch_size, max_patch_length, frames_per_patch * in_channels)
        x = self.in_proj(target)

        selection_mask = self.generate_mask(
            x, selection_rate=selection_rate, cluster_size=cluster_size
        )
        selection_mask = selection_mask.view(batch_size, max_patch_length, 1)
        x = x.masked_fill(selection_mask, 0)
        x = self.transformer_forward(x)
        # split cls token
        cls_output, output = torch.split(x, [1, max_patch_length], dim=-2)

        selected_output = []
        selected_target = []

        for x_output, x_target, mask in zip(output, target, selection_mask):
            x_output = x_output.masked_select(mask)
            x_output = x_output.view(-1, embedding_dim)
            x_target = x_target.masked_select(mask)
            x_target = x_target.view(-1, frames_per_patch * in_channels)
            selected_output.append(x_output)
            selected_target.append(x_target)

        output = nn.utils.rnn.pad_sequence(selected_output, batch_first=True)
        target = nn.utils.rnn.pad_sequence(selected_target, batch_first=True)
        length = selection_mask.sum(dim=(-2, -1))

        output = torch.cat([cls_output, output], dim=-2)

        return output, target, length

    @staticmethod
    def generate_mask(
        input: torch.Tensor,
        selection_rate: float = 0.15,
        cluster_size: int = 3,
    ) -> torch.BoolTensor:
        batch_size, max_length, _ = input.size()
        device = input.device

        num_masked_tokens = int(max_length * selection_rate)
        selection_mask = []

        for _ in range(batch_size):
            padding_index = torch.randperm(max_length)[:num_masked_tokens]
            mask = torch.zeros((max_length,), device=device)
            mask.scatter_(0, padding_index, 1)
            selection_mask.append(mask)

        selection_mask = torch.stack(selection_mask, dim=0)
        selection_mask = selection_mask.view(batch_size, 1, max_length)

        padding = cluster_size - 1
        padding_left = padding // 2
        padding_right = padding - padding_left

        selection_mask = selection_mask.expand(batch_size, cluster_size, max_length)
        selection_mask = F.fold(
            selection_mask, (1, max_length + cluster_size - 1), (1, cluster_size)
        )
        selection_mask = selection_mask.view(batch_size, -1)
        selection_mask = F.pad(selection_mask, (-padding_left, -padding_right))
        selection_mask = selection_mask > 0
        selection_mask = selection_mask.to(device)

        return selection_mask


class AudioTransformerMaskedPatchModel(nn.Module):
    """Audio transformer to train backbone as masked patch model."""

    def __init__(
        self,
        backbone: AudioTransformerMaskedPatchModelBackbone,
        classifier: nn.Module,
        reconstructor: nn.Module,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.classifier = classifier
        self.reconstructor = reconstructor

        assert self.backbone.batch_first, "Only batch_first=True is supported."

    def forward(
        self,
        input: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.LongTensor]:
        x, target, length = self.backbone(input)
        max_length = x.size(-2) - 1

        # remove cls token
        # NOTE: self.backbone concatenates cls token, so splitting it here is
        #       a little bit redundant.
        _, x = torch.split(x, [1, max_length], dim=-2)
        classified = self.classifier(x)
        reconstructed = self.reconstructor(x)

        return (classified, reconstructed), target, length


class AudioTransformerTower(nn.Module):
    def __init__(
        self,
        backbone: AudioTransformerBackbone,
        aggregator: nn.Module,
    ) -> None:
        super().__init__()

        assert isinstance(backbone, AudioTransformerBackbone)
        assert not isinstance(backbone, AudioTransformerMaskedPatchModelBackbone)

        self.backbone = backbone
        self.aggregator = aggregator

    def forward(
        self,
        input: torch.Tensor,
        length: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        x = self.backbone(input, length=length)
        output = self.aggregator(x)

        return output
