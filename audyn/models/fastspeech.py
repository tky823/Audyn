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
            ``length_regulator`` should take following values:

            - h_src (torch.Tensor): latent representation of source feature of shape
                (batch_size, src_length, num_features) or (src_length, batch_size, num_features).
            - duration (torch.LongTensor): linear duration of shape (batch_size, src_length) or
                (src_length, batch_size).
            - max_length (int, optional): max_length of each src token.

            and returns following values:

            - torch.Tensor: Expanded sequence of shape
                (batch_size, scaled_src_length, num_features) or
                (scaled_src_length, batch_size, num_features).
            - torch.LongTensor: Scaled duration of shape (batch_size, src_length) or
                (src_length, batch_size).

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
    ) -> None:
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
            src (torch.Tensor): Text input of shape (batch_size, src_length)
                if ``batch_first=True``. Otherwise, (src_length, batch_size).
            duration (torch.LongTensor): Duration of source of shape (batch_size, src_length)
                if ``batch_first=True``. Otherwise, (src_length, batch_size).
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

            if not self.batch_first:
                # NOTE: We assume encoder supports (batch_size, tgt_length)
                #       even when batch_first = False.
                src_key_padding_mask = src_key_padding_mask.permute(1, 0)

        _validate_padding_mask_shape(
            src,
            padding_mask=src_key_padding_mask,
            batch_first=self.batch_first,
        )
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

        if self.batch_first:
            tgt_length = linear_est_duration.sum(dim=1)
        else:
            tgt_length = linear_est_duration.sum(dim=0)

        max_tgt_length = torch.max(tgt_length)
        tgt_key_padding_mask = self.create_padding_mask(tgt_length, max_length=max_tgt_length)

        _validate_padding_mask_shape(
            h_tgt,
            padding_mask=tgt_key_padding_mask,
            batch_first=self.batch_first,
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
    def create_padding_mask(
        length: torch.Tensor, max_length: Optional[int] = None
    ) -> torch.BoolTensor:
        """Create padding mask for target sequence.

        Args:
            length (torch.Tensor): Length (i.e. sum of durations) of shape (batch_size,)
            max_length (int, optional): Max value of lengths.

        Returns:
            torch.BoolTensor: padding mask of shape (batch_size, max_duration).

        """
        # Allocation is required for mps
        indices = torch.arange(max_length).to(length.device)
        padding_mask = indices >= length.unsqueeze(dim=-1)

        return padding_mask


class MultiSpeakerFastSpeech(FastSpeech):
    """FastSpeech for multi-speaker.

    Args:
        encoder (nn.Module): Encoder to transform text embeddings to
            token-lavel latent representations.
        decoder (nn.Module): Decoder to transform frame-level latent representations.
        duration_predictor (nn.Module): Duration predictor to predict duration
            from token-level latent representations.
        length_regulator (nn.Module): Length regulator to control scale of length.
            ``length_regulator`` should take following values:

            - h_src (torch.Tensor): latent representation of source feature of shape
                (batch_size, src_length, num_features) or (src_length, batch_size, num_features).
            - duration (torch.LongTensor): linear duration of shape (batch_size, src_length) or
                (src_length, batch_size).
            - max_length (int, optional): max_length of each src token.

            and returns following values:

            - torch.Tensor: Expanded sequence of shape
                (batch_size, scaled_src_length, num_features) or
                (scaled_src_length, batch_size, num_features).
            - torch.LongTensor: Scaled duration of shape (batch_size, src_length) or
                (src_length, batch_size).

        speaker_encoder (nn.Module): Speaker encoder.
        blend_type (str): Blend type. ``add`` and ``mul`` are supported.
        batch_first (bool): If ``True``, tensors are treated as (batch_size, length, num_features).
            Otherwise, treated as (length, batch_size, num_features). Default: ``False``.

    """

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
        """Forward pass of MultiSpeakerFastSpeech.

        Args:
            src (torch.Tensor): Text input of shape (batch_size, src_length)
                if ``batch_first=True``. Otherwise, (src_length, batch_size).
            speaker (torch.Tensor): Speaker-like feature of shape (batch_size, *).
            duration (torch.LongTensor): Duration of source of shape (batch_size, src_length)
                if ``batch_first=True``. Otherwise, (src_length, batch_size).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Estimated feature of shape (batch_size, out_channels, tgt_length).
                - torch.Tensor: Estimated duration in log-domain of shape (batch_size, src_length).

        """
        if duration is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = duration == 0

            if not self.batch_first:
                # NOTE: We assume encoder supports (batch_size, tgt_length)
                #       even when batch_first = False.
                src_key_padding_mask = src_key_padding_mask.permute(1, 0)

        _validate_padding_mask_shape(
            src,
            padding_mask=src_key_padding_mask,
            batch_first=self.batch_first,
        )
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

        if self.batch_first:
            tgt_length = linear_est_duration.sum(dim=1)
        else:
            tgt_length = linear_est_duration.sum(dim=0)

        max_tgt_length = torch.max(tgt_length)
        tgt_key_padding_mask = self.create_padding_mask(tgt_length, max_length=max_tgt_length)

        h_tgt = self.blend_embeddings(h_tgt, spk_emb)

        _validate_padding_mask_shape(
            h_tgt,
            padding_mask=tgt_key_padding_mask,
            batch_first=self.batch_first,
        )
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

        _validate_padding_mask_shape(
            x,
            padding_mask=src_key_padding_mask,
            batch_first=self.batch_first,
        )

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

        _validate_padding_mask_shape(
            x,
            padding_mask=tgt_key_padding_mask,
            batch_first=self.batch_first,
        )

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
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Forward pass of length regulator.

        Args:
            sequence (torch.Tensor): Input sequence of shape (batch_size, length, *)
                if ``self.batch_first=True``. Otherwise, (length, batch_size, *).
            duration (torch.LongTensor): Duration of each input token of shape
                (batch_size, length) if ``self.batch_first=True``. Otherwise, (length, batch_size).
            expand_scale (float): Parameter to control scale of duration. Default: ``1``.
            padding_mask (torch.BoolTensor, optional): Padding mask
                of shape (batch_size, length).
            max_length (int, optional): Maximum length of output sequence
                to avoid out-of-memory.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Expanded sequence of shape (batch_size, expanded_length, *)
                    if ``self.batch_first=True``. Otherwise, (length, batch_size, *).
                - torch.Tensor: Scaled duration of shape (batch_size, length, *)
                    if ``self.batch_first=True``. Otherwise, (length, batch_size).

        """
        pad_value = self.pad_value
        batch_first = self.batch_first

        scaled_duration = expand_scale * duration
        scaled_duration = scaled_duration.float()
        scaled_duration = torch.clip(
            scaled_duration,
            min=self.min_duration,
            max=self.max_duration,
        )
        scaled_duration = torch.round(scaled_duration)
        scaled_duration = scaled_duration.long()

        # set 0 to cancel out self.min_duration
        zero_padding_mask = duration == 0
        scaled_duration = scaled_duration.masked_fill(zero_padding_mask, 0)

        if padding_mask is not None:
            # Unsqueeze padding mask
            padding_mask_shape = sequence.size()[:2]
            num_feature_dims = sequence.dim() - len(padding_mask_shape)
            unsqueezed_padding_mask_shape = padding_mask_shape + (1,) * num_feature_dims
            unsqueezed_padding_mask = padding_mask.view(*unsqueezed_padding_mask_shape)
            sequence = sequence.masked_fill(unsqueezed_padding_mask, value=self.pad_value)

        expanded_sequence = expand_by_duration(
            sequence,
            scaled_duration,
            pad_value=pad_value,
            batch_first=batch_first,
            max_length=max_length,
        )

        return expanded_sequence, scaled_duration


def _get_clones(module, N) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _validate_padding_mask_shape(
    input: torch.Tensor,
    padding_mask: Optional[torch.BoolTensor] = None,
    batch_first: Optional[bool] = None,
) -> None:
    """Validate shape of padding mask is compatible with that of input depending on batch_first."""
    if padding_mask is None:
        return

    if batch_first is None:
        raise ValueError("Specify batch_first.")

    if batch_first:
        assert input.size(0) == padding_mask.size(0)
        assert input.size(1) == padding_mask.size(1)
    else:
        assert input.size(1) == padding_mask.size(0)
        assert input.size(0) == padding_mask.size(1)
