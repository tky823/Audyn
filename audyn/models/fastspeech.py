import copy
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..modules.fastspeech import FFTrBlock
from ..utils.alignment import expand_by_duration

__all__ = ["FastSpeech", "MultiSpeakerFastSpeech", "LengthRegulator"]


class FastSpeech(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        length_regulator: nn.Module,
        batch_first: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.length_regulator = length_regulator
        self.decoder = decoder

        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if duration is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = duration == 0

        h_src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        h_tgt, log_est_duration = self.length_regulator(
            h_src,
            duration=duration,
            max_length=max_length,
        )

        if duration is None:
            if hasattr(self.length_regulator, "min_duration"):
                min_duration = self.length_regulator.min_duration
            else:
                min_duration = None

            if hasattr(self.length_regulator, "max_duration"):
                max_duration = self.length_regulator.max_duration
            else:
                max_duration = None

            linear_est_duration = _transform_log_duration(
                log_est_duration, min_duration=min_duration, max_duration=max_duration
            )

            if src_key_padding_mask is not None:
                linear_est_duration = linear_est_duration.masked_fill(src_key_padding_mask, 0)
        else:
            linear_est_duration = duration

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
        if duration is None:
            src_key_padding_mask = None
        else:
            src_key_padding_mask = duration == 0

        spk_emb = self.speaker_encoder(speaker)
        h_src = self.encoder(src, src_key_padding_mask=src_key_padding_mask)
        h_src = self.blend_embeddings(h_src, spk_emb)
        h_tgt, log_est_duration = self.length_regulator(
            h_src,
            duration=duration,
            max_length=max_length,
        )

        if duration is None:
            if hasattr(self.length_regulator, "min_duration"):
                min_duration = self.length_regulator.min_duration
            else:
                min_duration = None

            if hasattr(self.length_regulator, "max_duration"):
                max_duration = self.length_regulator.max_duration
            else:
                max_duration = None

            linear_est_duration = _transform_log_duration(
                log_est_duration, min_duration=min_duration, max_duration=max_duration
            )

            if src_key_padding_mask is not None:
                linear_est_duration = linear_est_duration.masked_fill(src_key_padding_mask, 0)
        else:
            linear_est_duration = duration

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

        if self.norm is not None:
            x = self.norm(x)

            if src_key_padding_mask is not None:
                output = x
            else:
                output = self.apply_mask(x, src_key_padding_mask=src_key_padding_mask)
        else:
            output = x

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
        duration_predictor: nn.Module,
        pad_value: float = 0,
        batch_first: bool = False,
        min_duration: int = 1,
        max_duration: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.duration_predictor = duration_predictor

        self.pad_value = pad_value
        self.batch_first = batch_first
        self.min_duration = min_duration
        self.max_duration = max_duration

    def forward(
        self,
        sequence: torch.Tensor,
        duration: Optional[torch.LongTensor] = None,
        expand_ratio: float = 1,
        padding_mask: Optional[torch.BoolTensor] = None,
        max_length: Optional[int] = None,
    ):
        pad_value = self.pad_value
        batch_first = self.batch_first

        log_est_duration = self.duration_predictor(sequence, padding_mask=padding_mask)

        if duration is None:
            linear_est_duration = _transform_log_duration(
                log_est_duration, min_duration=self.min_duration, max_duration=self.max_duration
            )

            if padding_mask is not None:
                linear_est_duration = linear_est_duration.masked_fill(padding_mask, 0)

            alignment = expand_by_duration(
                sequence,
                linear_est_duration.long(),
                pad_value=pad_value,
                batch_first=batch_first,
                max_length=max_length,
            )
        else:
            duration = torch.round(expand_ratio * duration.float())
            alignment = expand_by_duration(
                sequence,
                duration.long(),
                pad_value=pad_value,
                batch_first=batch_first,
                max_length=max_length,
            )

        return alignment, log_est_duration


def _get_clones(module, N) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _transform_log_duration(
    log_duration: torch.Tensor, min_duration: int = 1, max_duration: Optional[int] = None
) -> torch.Tensor:
    """Transform log-duration to linear-duration.

    Args:
        log_duration (torch.Tensor): Duration in log-domain.
        min_duration (int): Min duration in linear domain. Default: ``1``.
        max_duration (int, optional): Max duration in linear domain. Default: ``None``.

    Returns:
        torch.Tensor: Duration in linear-domain.

    """
    linear_duration = torch.exp(log_duration)
    linear_duration = torch.round(linear_duration.detach())
    linear_duration = torch.clip(linear_duration, min=min_duration, max=max_duration)

    return linear_duration
