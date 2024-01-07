from typing import Optional, Tuple

import torch
import torch.nn as nn

from audyn.models.soundstream import SoundStream
from audyn.models.text_to_wave import CascadeTextToWave

__all__ = ["VALLE"]


class VALLE(nn.Module):
    def __init__(
        self,
        text_embedding: nn.Embedding,
        acoustic_embedding: nn.Embedding,
        text_positional_encoding: nn.Module,
        acoustic_positional_encoding: nn.Module,
        decoder: nn.Module,
        out_proj: nn.Linear,
        text_pad_idx: int = 0,
        text_eos_idx: int = 1,
        acoustic_pad_idx: int = 0,
        acoustic_eos_idx: int = 1,
        channels_last: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(text_embedding, nn.Embedding):
            raise ValueError("text_embedding should be a subclass of nn.Embedding.")

        if not isinstance(acoustic_embedding, nn.Embedding):
            raise ValueError("acoustic_embedding should be a subclass of nn.Embedding.")

        assert (
            acoustic_embedding.padding_idx == acoustic_pad_idx
        ), "padding_idx of acoustic_embedding should be 0."
        assert channels_last, "Only channels_last=True is supported."

        self.text_embedding = text_embedding
        self.acoustic_embedding = acoustic_embedding
        self.text_positional_encoding = text_positional_encoding
        self.acoustic_positional_encoding = acoustic_positional_encoding
        self.decoder = decoder
        self.out_proj = out_proj
        self.text_pad_idx = text_pad_idx
        self.text_eos_idx = text_eos_idx
        self.acoustic_pad_idx = acoustic_pad_idx
        self.acoustic_eos_idx = acoustic_eos_idx

    def forward(
        self,
        text: torch.LongTensor,
        acoustic: torch.LongTensor,
        text_length: torch.LongTensor = None,
        acoustic_length: torch.LongTensor = None,
    ) -> torch.Tensor:
        """Forward pass of VALLE.

        Args:
            text (torch.LongTensor): Text tokens of shape (batch_size, max_text_length).
            acoustic (torch.LongTensor): Acoustic tokens
                of shape (batch_size, max_acoustic_length).
            text_length (torch.LongTensor, optional): Lengths of text tokens
                of shape (batch_size,).
            acoustic_length (torch.LongTensor, optional): Lengths of acoustic tokens
                of shape (batch_size,).

        Returns:
            torch.Tensor: Logit of shape (batch_size, max_acoustic_length, codebook_size + 1),
                where additional token (index 0) means EOS of acoustic tokens.

        """
        batch_size = text.size(0)

        assert (
            acoustic.size(0) == batch_size
        ), "Batch size of acoustic tokens is inconsistent with that of text tokens."

        x_text = self.text_embedding(text)
        x_acoustic = self.acoustic_embedding(acoustic)
        x_text = self.text_positional_encoding(x_text)
        x_acoustic = self.acoustic_positional_encoding(x_acoustic)

        if text_length is None:
            max_text_length = text.size(-1)
            text_length = torch.full(
                (batch_size,),
                fill_value=max_text_length,
                dtype=torch.long,
                device=text.device,
            )

        if acoustic_length is None:
            max_acoustic_length = text.size(-1)
            acoustic_length = torch.full(
                (batch_size,),
                fill_value=max_acoustic_length,
                dtype=torch.long,
                device=text.device,
            )

        x = self.cat_features(
            x_text,
            x_acoustic,
            text_length=text_length,
            acoustic_length=acoustic_length,
        )
        causal_padding_mask = self.generate_causal_mask(x.size(1), device=x.device)
        full_padding_mask = self.generate_src_key_padding_mask(
            text_length + acoustic_length, device=x.device
        )
        acoustic_padding_mask = self.generate_src_key_padding_mask(
            acoustic_length, device=x.device
        )
        x = self.decoder(x, mask=causal_padding_mask, src_key_padding_mask=full_padding_mask)

        # including estimated <EOS> of acoustic tokens
        _, x_acoustic = self.split_features(
            x, text_length=text_length, acoustic_length=acoustic_length
        )
        logit = self.out_proj(x_acoustic)
        logit = logit.masked_fill(acoustic_padding_mask.unsqueeze(dim=-1), 0)

        return logit

    @torch.no_grad()
    def inference(
        self, text: torch.LongTensor, max_length: Optional[int] = None
    ) -> torch.LongTensor:
        batch_size = text.size(0)
        device = text.device

        assert batch_size == 1, f"Batch size should be 1, but {batch_size} is given."

        x_text = self.text_embedding(text)
        x_acoustic = torch.zeros(
            (batch_size, 0, x_text.size(-1)), device=device, dtype=x_text.dtype
        )
        output = torch.zeros((batch_size, 0), device=device, dtype=torch.long)

        frame_idx = 0

        while True:
            x = torch.cat([x_text, x_acoustic], dim=1)
            max_full_length = x.size(1)
            positions = torch.arange(max_full_length, device=device)
            causal_padding_mask = positions > positions.unsqueeze(dim=-1)
            x = self.decoder(x, mask=causal_padding_mask)
            _, x_last_acoustic = torch.split(x, [max_full_length - 1, 1], dim=1)
            last_logit = self.out_proj(x_last_acoustic)
            last_output = torch.softmax(last_logit, dim=-1)
            last_output = torch.distributions.Categorical(last_output).sample()

            if last_output.item() == 0:
                break

            output = torch.cat([output, last_output], dim=1)

            frame_idx += 1

            if max_length is not None and frame_idx >= max_length:
                break

            acoustic_last = self.acoustic_embedding(last_output)
            x_acoustic = torch.cat([x_acoustic, acoustic_last], dim=1)

        output = output - 1

        return output

    def cat_features(
        self,
        text: torch.Tensor,
        acoustic: torch.Tensor,
        text_length: torch.LongTensor,
        acoustic_length: torch.LongTensor,
    ) -> torch.Tensor:
        x = []

        max_text_length = text.size(1)
        max_acoustic_length = acoustic.size(1)

        for _text, _acoustic, _text_length, _acoustic_length in zip(
            text, acoustic, text_length, acoustic_length
        ):
            _text_length, _acoustic_length = _text_length.item(), _acoustic_length.item()
            _text, _ = torch.split(_text, [_text_length, max_text_length - _text_length], dim=0)
            _acoustic, _ = torch.split(
                _acoustic, [_acoustic_length, max_acoustic_length - _acoustic_length], dim=0
            )
            _x = torch.cat([_text, _acoustic], dim=0)
            x.append(_x)

        output = nn.utils.rnn.pad_sequence(x, batch_first=True)

        return output

    def split_features(
        self,
        input: torch.Tensor,
        text_length: torch.LongTensor,
        acoustic_length: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        max_length = input.size(1)
        text = []
        acoustic = []

        for _input, _text_length, _acoustic_length in zip(input, text_length, acoustic_length):
            _text_length, _acoustic_length = _text_length.item(), _acoustic_length.item()
            _text, _acoustic, _ = torch.split(
                _input,
                [
                    _text_length - 1,
                    _acoustic_length,
                    max_length - (_text_length + _acoustic_length - 1),
                ],
                dim=0,
            )
            text.append(_text)
            acoustic.append(_acoustic)

        text = nn.utils.rnn.pad_sequence(text, batch_first=True)
        acoustic = nn.utils.rnn.pad_sequence(acoustic, batch_first=True)

        return text, acoustic

    @staticmethod
    def generate_causal_mask(
        length: int,
        device: torch.device = None,
    ) -> torch.BoolTensor:
        factory_kwargs = {
            "device": device,
            "dtype": torch.long,
        }
        positions = torch.arange(length, **factory_kwargs)
        padding_mask = positions > positions.unsqueeze(dim=-1)

        return padding_mask

    @staticmethod
    def generate_src_key_padding_mask(
        length: torch.LongTensor,
        device: torch.device = None,
    ) -> torch.BoolTensor:
        factory_kwargs = {
            "device": device,
            "dtype": torch.long,
        }
        max_length = torch.max(length)
        positions = torch.arange(max_length, **factory_kwargs)
        src_key_padding_mask = positions >= length.unsqueeze(dim=-1)

        return src_key_padding_mask


class VALLETTS(CascadeTextToWave):
    def __init__(
        self,
        text_to_feat: VALLE,
        feat_to_wave: SoundStream,
    ) -> None:
        super().__init__(text_to_feat, feat_to_wave)

        self.text_to_feat: VALLE
        self.feat_to_wave: SoundStream

    @torch.no_grad()
    def inference(self, text: torch.LongTensor, max_length: Optional[int] = None) -> torch.Tensor:
        """

        Args:
            text (torch.LongTensor): Text tokens of shape (batch_size, max_text_length).
            max_length (int, optional): Max length of acoustic features.

        Returns:
            torch.Tensor: Estimated waveform of shape (batch_size, out_channels, max_timesteps).

        """
        feat_to_wave = self.feat_to_wave
        codebook = feat_to_wave.vector_quantizer.codebooks[0]

        estimated_indices = self.text_to_feat.inference(text, max_length=max_length)
        quantized = codebook(estimated_indices)
        quantized = quantized.permute(0, 2, 1)
        output = feat_to_wave.decode(quantized, stage_wise=False)

        return output

    @property
    def down_scale(self) -> Optional[int]:
        return self.feat_to_wave.down_scale
