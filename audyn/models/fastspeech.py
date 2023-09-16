import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..modules.fastspeech import FFTrBlock
from ..utils.alignment import expand_by_duration
from ..utils.duration import transform_log_duration

__all__ = ["FastSpeech", "MultiSpeakerFastSpeech", "LengthRegulator"]


class FastSpeech(nn.Module):
    """FastSpeech proposed in 'FastSpeech: Fast, robust and controllable text to speech.'

    Args:
        encoder (nn.Module): Encoder to transform text embeddings to
            token-lavel latent representations.
        decoder (nn.Module): Decoder to transform frame-level latent representations.
        duration_predictor (nn.Module): Duration predictor to predict duration
            from token-level latent representations.
        length_regulator (nn.Module): Length regulator to control scale of length.
        batch_first (bool): If ``True``, tensors are treated as (batch_size, length, num_features).
            Otherwise, treated as (length, batch_size, num_features). Default: ``False``.

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        duration_predictor: nn.Module,
        length_regulator: nn.Module,
        batch_first: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.length_regulator = length_regulator
        self.decoder = decoder

        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of FastSpeech.

        Args:
            src (torch.Tensor): Text input of shape (batch_size, src_length).
            duration (torch.LongTensor): Duration of source of shape (batch_size, src_length).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length * src_length.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Estimated feature of shape (batch_size, out_channels, tgt_length).
                - torch.Tensor: Estimated duration in log-domain of shape (batch_size, src_length).

        """
        if duration is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = duration == 0

        h_src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        log_est_duration = self.duration_predictor(h_src)

        if duration is None:
            # Estimated duration is used.
            if hasattr(self.length_regulator, "min_duration"):
                min_duration = self.length_regulator.min_duration
            else:
                min_duration = None

            if hasattr(self.length_regulator, "max_duration"):
                max_duration = self.length_regulator.max_duration
            else:
                max_duration = None

            linear_est_duration = transform_log_duration(
                log_est_duration,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if src_key_padding_mask is not None:
                linear_est_duration = linear_est_duration.masked_fill(src_key_padding_mask, 0)
        else:
            linear_est_duration = duration

        h_tgt, _ = self.length_regulator(
            h_src,
            duration=linear_est_duration,
            max_length=max_length,
        )

        tgt_duration = linear_est_duration.sum(dim=1)
        max_tgt_duration = torch.max(tgt_duration)
        tgt_key_padding_mask = self.create_duration_padding_mask(
            tgt_duration, max_duration=max_tgt_duration
        )

        output = self.decoder(h_tgt, tgt_key_padding_mask=tgt_key_padding_mask)

        return output, log_est_duration

    @torch.no_grad()
    def inference(
        self,
        src: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            src,
            duration=duration,
            max_length=max_length,
        )

    @staticmethod
    def create_duration_padding_mask(
        duration: torch.Tensor, max_duration: Optional[int] = None
    ) -> torch.BoolTensor:
        """Create padding mask for duration.

        Args:
            duration (torch.Tensor): Duration of shape (batch_size,)
            max_duration (int, optional): Max value of durations.

        Returns:
            torch.BoolTensor of padding mask.
            The shape is (batch_size, max_duration)

        """
        # Allocation is required for mps
        indices = torch.arange(max_duration).to(duration.device)
        padding_mask = indices >= duration.unsqueeze(dim=-1)

        return padding_mask


class MultiSpeakerFastSpeech(FastSpeech):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        duration_predictor: nn.Module,
        length_regulator: nn.Module,
        speaker_encoder: nn.Module,
        blend_type: str = "add",
        batch_first: bool = False,
    ) -> None:
        # Use nn.Module.__init__ just for readability of
        # print(MultiSpeakerFastSpeech(...))
        super(FastSpeech, self).__init__()

        self.speaker_encoder = speaker_encoder
        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.length_regulator = length_regulator
        self.decoder = decoder

        self.blend_type = blend_type
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        speaker: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of FastSpeech.

        Args:
            src (torch.Tensor): Text input of shape (batch_size, src_length).
            speaker (torch.Tensor): Speaker-like feature of shape (batch_size, *).
            duration (torch.LongTensor): Duration of source of shape (batch_size, src_length).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length * src_length.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Estimated feature of shape (batch_size, out_channels, tgt_length).
                - torch.Tensor: Estimated duration in log-domain of shape (batch_size, src_length).

        """
        if duration is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = duration == 0

        spk_emb = self.speaker_encoder(speaker)
        h_src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        h_src = self.blend_embeddings(h_src, spk_emb)
        log_est_duration = self.duration_predictor(h_src)

        if duration is None:
            # Estimated duration is used.
            if hasattr(self.length_regulator, "min_duration"):
                min_duration = self.length_regulator.min_duration
            else:
                min_duration = None

            if hasattr(self.length_regulator, "max_duration"):
                max_duration = self.length_regulator.max_duration
            else:
                max_duration = None

            linear_est_duration = transform_log_duration(
                log_est_duration,
                min_duration=min_duration,
                max_duration=max_duration,
            )

            if src_key_padding_mask is not None:
                linear_est_duration = linear_est_duration.masked_fill(src_key_padding_mask, 0)
        else:
            linear_est_duration = duration

        h_tgt, _ = self.length_regulator(
            h_src,
            duration=linear_est_duration,
            max_length=max_length,
        )

        tgt_duration = linear_est_duration.sum(dim=1)
        max_tgt_duration = torch.max(tgt_duration)
        tgt_key_padding_mask = self.create_duration_padding_mask(
            tgt_duration, max_duration=max_tgt_duration
        )

        h_tgt = self.blend_embeddings(h_tgt, spk_emb)
        output = self.decoder(h_tgt, tgt_key_padding_mask=tgt_key_padding_mask)

        return output, log_est_duration

    @torch.no_grad()
    def inference(
        self,
        src: torch.Tensor,
        speaker: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(
            src,
            speaker,
            duration=duration,
            max_length=max_length,
        )

    def blend_embeddings(self, latent: torch.Tensor, spk_emb: torch.Tensor) -> torch.Tensor:
        """Blend latent and speaker embeddings.

        Args:
            latent (torch.Tensor): Latent feature of shape (batch_size, length, num_features)
                if self.batch_first = True. Otherwise, (length, batch_size, num_features).
            spk_emb (torch.Tensor): Speaker embedding of shape (batch_size, num_features)

        Returns:
            torch.Tensor of blended embeddings of shape (batch_size, length, num_features)
                if self.batch_first = True. Otherwise, (length, batch_size, num_features).

        """
        if self.batch_first:
            spk_emb = spk_emb.unsqueeze(dim=1)

        if self.blend_type == "add":
            return latent + spk_emb
        elif self.blend_type == "mul":
            return latent * spk_emb
        else:
            raise ValueError(f"blend_type={self.blend_type} is not supported.")


class Encoder(nn.Module):
    def __init__(
        self,
        word_embedding: nn.Module,
        positional_encoding: nn.Module,
        encoder_layer: nn.Module,
        num_layers: int,
        batch_first: bool = False,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.word_embedding = word_embedding
        self.positional_encoding = positional_encoding
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = norm

        if isinstance(encoder_layer, FFTrBlock):
            self.required_kwargs = {"need_weights": False}
        else:
            self.required_kwargs = {}

        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of feed-forward transformer block for FastSpeech.

        Args:
            src (torch.LongTensor): Word indices of shape (\*).
            mask (torch.Tensor): Attention mask for source of shape
                (src_length, src_length) or (batch_size * num_heads, src_length, src_length).
            src_key_padding_mask (torch.Tensor): Padding mask of shape (src_length,)
                or (batch_size, src_length).

        Returns:
            torch.Tensor: Encoded input of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise, (src_length, batch_size, embed_dim).

        """
        x = self.word_embedding(src)
        x = self.positional_encoding(x)

        for module in self.layers:
            x = module(
                x,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                **self.required_kwargs,
            )

        if src_key_padding_mask is not None:
            x = self.apply_mask(x, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        if src_key_padding_mask is None:
            output = x
        else:
            output = self.apply_mask(x, src_key_padding_mask=src_key_padding_mask)

        return output

    def apply_mask(
        self, input: torch.Tensor, src_key_padding_mask: torch.BoolTensor
    ) -> torch.Tensor:
        padding_mask = src_key_padding_mask.unsqueeze(dim=-1)

        if self.batch_first:
            output = input.masked_fill(padding_mask, 0)
        else:
            if src_key_padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(dim=-1)
                output = input.masked_fill(padding_mask, 0)
            elif src_key_padding_mask.dim() == 2:
                output = input.masked_fill(padding_mask.swapaxes(0, 1), 0)
            else:
                raise ValueError(
                    "src_key_padding_mask is expected to be 1 or 2D tensor,"
                    f"but given {src_key_padding_mask.dim()}."
                )

        return output


class Decoder(nn.Module):
    def __init__(
        self,
        positional_encoding: nn.Module,
        decoder_layer: nn.Module,
        fc_layer: nn.Module,
        num_layers: int,
        batch_first: bool = False,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.positional_encoding = positional_encoding
        self.layers = _get_clones(decoder_layer, num_layers)
        self.norm = norm
        self.fc_layer = fc_layer

        if isinstance(decoder_layer, FFTrBlock):
            self.required_kwargs = {"need_weights": False}
        else:
            raise ValueError(
                f"Only FFTrBlock is supported, but {decoder_layer.__class__.__name__} is given."
            )

        if isinstance(fc_layer, nn.Linear):
            self.required_kwargs = {"need_weights": False}
        else:
            raise ValueError(
                f"Only nn.Linear is supported, but {decoder_layer.__class__.__name__} is given."
            )

        self.num_layers = num_layers
        self.batch_first = batch_first

    def forward(
        self,
        tgt: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        tgt_key_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        r"""Forward pass of feed-forward transformer block for FastSpeech.

        Args:
            tgt (torch.Tensor): Target embedding of shape (batch_size, tgt_length, embed_dim)
                if ``batch_first=True``, otherwise (tgt_length, batch_size, embed_dim).
            mask (torch.Tensor): Attention mask for target of shape
                (tgt_length, tgt_length) or (batch_size * num_heads, tgt_length, tgt_length).
            tgt_key_padding_mask (torch.Tensor): Padding mask of shape (tgt_length,)
                or (batch_size, tgt_length).

        Returns:
            torch.Tensor: Encoded input of shape (batch_size, tgt_length, out_features)
                if ``batch_first=True``. Otherwise, (tgt_length, batch_size, out_features).

        """
        x = self.positional_encoding(tgt)

        for module in self.layers:
            x = module(
                x,
                src_mask=mask,
                src_key_padding_mask=tgt_key_padding_mask,
                **self.required_kwargs,
            )

        if tgt_key_padding_mask is not None:
            x = self.apply_mask(x, tgt_key_padding_mask=tgt_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)

        if tgt_key_padding_mask is not None:
            x = self.apply_mask(x, tgt_key_padding_mask=tgt_key_padding_mask)

        x = self.fc_layer(x)

        if tgt_key_padding_mask is None:
            output = x
        else:
            output = self.apply_mask(x, tgt_key_padding_mask=tgt_key_padding_mask)

        return output

    def apply_mask(
        self, input: torch.Tensor, tgt_key_padding_mask: torch.BoolTensor
    ) -> torch.Tensor:
        padding_mask = tgt_key_padding_mask.unsqueeze(dim=-1)

        if self.batch_first:
            output = input.masked_fill(padding_mask, 0)
        else:
            if tgt_key_padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(dim=-1)
                output = input.masked_fill(padding_mask, 0)
            elif tgt_key_padding_mask.dim() == 2:
                output = input.masked_fill(padding_mask.swapaxes(0, 1), 0)
            else:
                raise ValueError(
                    "tgt_key_padding_mask is expected to be 1 or 2D tensor,"
                    f"but given {tgt_key_padding_mask.dim()}."
                )

        return output


class LengthRegulator(nn.Module):
    def __init__(
        self,
        pad_value: float = 0,
        batch_first: bool = False,
        min_duration: int = 1,
        max_duration: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.pad_value = pad_value
        self.batch_first = batch_first
        self.min_duration = min_duration
        self.max_duration = max_duration

    def forward(
        self,
        sequence: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        expand_scale: float = 1,
        padding_mask: Optional[torch.BoolTensor] = None,
        max_length: Optional[int] = None,
    ):
        """Forward pass of length regulator.

        Args:
            sequence (torch.Tensor): Input sequence of shape (batch_size, length, *)
                if ``self.batch_first=True``. Otherwise, (length, batch_size, *).
            duration (torch.LongTensor): Duration of each input token
                of shape (batch_size, length).
            expand_scale (float): Parameter to control scale of duration. Default: ``1``.
            padding_mask (torch.BoolTensor, optional): Padding mask
                of shape (batch_size, length).
            max_length (int, optional): Maximum length of output sequence
                to avoid out-of-memory.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Expanded sequence.
                - torch.Tensor: Expanded duration.

        """
        pad_value = self.pad_value
        batch_first = self.batch_first

        expanded_duration = expand_scale * duration
        expanded_duration = torch.round(expanded_duration.float())
        expanded_duration = expanded_duration.long()

        if padding_mask is not None:
            # Unsqueeze padding mask
            padding_mask_shape = sequence.size()[:2]
            num_feature_dims = sequence.dim() - len(padding_mask_shape)
            unsqueezed_padding_mask_shape = padding_mask_shape + (1,) * num_feature_dims
            unsqueezed_padding_mask = padding_mask.view(*unsqueezed_padding_mask_shape)
            sequence = sequence.masked_fill(unsqueezed_padding_mask, value=self.pad_value)

        expanded_sequence = expand_by_duration(
            sequence,
            expanded_duration,
            pad_value=pad_value,
            batch_first=batch_first,
            max_length=max_length,
        )

        return expanded_sequence, expanded_duration


def _get_clones(module, N) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
