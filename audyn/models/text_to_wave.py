from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn

from ..utils.model import unwrap
from .fastspeech import FastSpeech
from .waveglow import WaveGlow
from .wavenet import WaveNet


class CascadeTextToWave(nn.Module):
    def __init__(
        self,
        text_to_feat: nn.Module,
        feat_to_wave: nn.Module,
        transform_middle: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.text_to_feat = text_to_feat
        self.feat_to_wave = feat_to_wave

        self.transform_middle = transform_middle

    def forward(
        self,
        text: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        feature = self.text_to_feat(text, max_length=max_length)

        if self.transform_middle is not None:
            feature = self.transform_middle(feature)

        waveform = self.feat_to_wave(feature)

        return waveform

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        max_length: Optional[int] = None,
    ) -> torch.Tensor:
        unwrapped_text_to_feat = unwrap(self.text_to_feat)
        unwrapped_transform_middle = unwrap(self.transform_middle)
        unwrapped_feat_to_wave = unwrap(self.feat_to_wave)

        if hasattr(unwrapped_text_to_feat, "inference"):
            feature = unwrapped_text_to_feat.inference(text, max_length=max_length)
        else:
            feature = unwrapped_text_to_feat(text, max_length=max_length)

        if unwrapped_transform_middle is not None:
            if hasattr(unwrapped_transform_middle, "inference"):
                feature = unwrapped_transform_middle.inference(feature)
            else:
                feature = unwrapped_transform_middle(feature)

        if hasattr(unwrapped_feat_to_wave, "inference"):
            waveform = unwrapped_feat_to_wave.inference(feature)
        else:
            waveform = unwrapped_feat_to_wave(feature)

        return waveform


class FastSpeechWaveNet(CascadeTextToWave):
    def __init__(
        self,
        text_to_feat: FastSpeech,
        feat_to_wave: WaveNet,
        transform_middle: nn.Module = None,
    ) -> None:
        self.text_to_feat: FastSpeech
        self.feat_to_wave: WaveNet

        super().__init__(
            text_to_feat,
            feat_to_wave,
            transform_middle=transform_middle,
        )

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError("forward pass is not supported.")

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        initial_state: torch.Tensor,
        global_conditioning: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference of FastSpeech + WaveNet.

        Args:
            text (torch.Tensor): Text sequence of shape (batch_size, text_length).
            initial_state (torch.Tensor): Initial state of shape
                (batch_size, 1) or (batch_size, in_channels, 1).
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).

        Returns:
            Tuple of tensors containing:

                - waveform (torch.Tensor): Estimated waveform of shape \
                    (batch_size, in_channels, waveform_length).
                - melspectrogram (torch.Tensor): Estimated Melspectrogram of shape \
                    (batch_size, n_mels, melspectrogram_length).
                - log_est_duration (torch.Tensor): Estimated log-duration of each token \
                    of shape (batch_size, text_length).

        """
        unwrapped_text_to_feat = unwrap(self.text_to_feat)
        unwrapped_transform_middle = unwrap(self.transform_middle)
        unwrapped_feat_to_wave = unwrap(self.feat_to_wave)

        if hasattr(unwrapped_transform_middle, "inference"):
            log_melspectrogram, log_est_duration = unwrapped_text_to_feat.inference(
                text, max_length=max_length
            )
        else:
            log_melspectrogram, log_est_duration = unwrapped_text_to_feat(
                text, max_length=max_length
            )

        if hasattr(unwrapped_transform_middle, "inference"):
            melspectrogram = unwrapped_transform_middle.inference(log_melspectrogram)
        else:
            melspectrogram = unwrapped_transform_middle(log_melspectrogram)

        if hasattr(unwrapped_feat_to_wave, "inference"):
            waveform = unwrapped_feat_to_wave.inference(
                initial_state,
                local_conditioning=melspectrogram,
                global_conditioning=global_conditioning,
            )
        else:
            waveform = unwrapped_feat_to_wave(
                initial_state,
                local_conditioning=melspectrogram,
                global_conditioning=global_conditioning,
            )

        return waveform, melspectrogram, log_est_duration

    def remove_weight_norm_(self) -> None:
        unwrapped_feat_to_wave = unwrap(self.feat_to_wave)

        if hasattr(unwrapped_feat_to_wave, "remove_weight_norm_"):
            unwrapped_feat_to_wave.remove_weight_norm_()


class FastSpeechWaveGlow(CascadeTextToWave):
    def __init__(
        self,
        text_to_feat: FastSpeech,
        feat_to_wave: WaveGlow,
        transform_middle: nn.Module = None,
    ) -> None:
        self.text_to_feat: FastSpeech
        self.feat_to_wave: WaveGlow

        super().__init__(
            text_to_feat,
            feat_to_wave,
            transform_middle=transform_middle,
        )

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError("forward pass is not supported.")

    @torch.no_grad()
    def inference(
        self,
        text: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        global_conditioning: Optional[torch.Tensor] = None,
        std: float = 1,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference of FastSpeech + WaveGlow.

        Args:
            text (torch.Tensor): Text sequence of shape (batch_size, text_length).
            noise (torch.Tensor): Noise sequence of shape
                (batch_size, in_channels, waveform_length), where in_channels is
                number channels in input of WaveGlow.
            local_conditioning (torch.Tensor, optional): Local conditioning of shape
                (batch_size, local_channels, local_length).
            global_conditioning (torch.Tensor, optional): Global conditioning of shape
                (batch_size, global_channels) or (batch_size, global_channels, 1).
            std (float): Standard deviation of noise. Default: 1.

        Returns:
            Tuple of tensors containing:

                - waveform (torch.Tensor): Estimated waveform of shape \
                    (batch_size, in_channels, waveform_length).
                - melspectrogram (torch.Tensor): Estimated Melspectrogram of shape \
                    (batch_size, n_mels, melspectrogram_length).
                - log_est_duration (torch.Tensor): Estimated log-duration of each token \
                    of shape (batch_size, text_length).

        """
        unwrapped_text_to_feat = unwrap(self.text_to_feat)
        unwrapped_transform_middle = unwrap(self.transform_middle)
        unwrapped_feat_to_wave = unwrap(self.feat_to_wave)

        if hasattr(unwrapped_text_to_feat, "inference"):
            log_melspectrogram, log_est_duration = unwrapped_text_to_feat.inference(
                text, max_length=max_length
            )
        else:
            log_melspectrogram, log_est_duration = unwrapped_text_to_feat(
                text, max_length=max_length
            )

        if hasattr(unwrapped_transform_middle, "inference"):
            melspectrogram = unwrapped_transform_middle.inference(log_melspectrogram)
        else:
            melspectrogram = unwrapped_transform_middle(log_melspectrogram)

        in_channels = unwrapped_feat_to_wave.in_channels
        upsampled_melspectrogram = unwrapped_feat_to_wave.upsample(melspectrogram)
        batch_size, _, length = upsampled_melspectrogram.size()

        if noise is None:
            noise = std * torch.randn(
                (batch_size, in_channels, length),
                dtype=upsampled_melspectrogram.dtype,
                device=upsampled_melspectrogram.device,
            )
        else:
            noise = std * noise

        if hasattr(unwrapped_feat_to_wave, "inference"):
            waveform = unwrapped_feat_to_wave.inference(
                noise,
                local_conditioning=melspectrogram,
                global_conditioning=global_conditioning,
            )
        else:
            waveform = unwrapped_feat_to_wave(
                noise,
                local_conditioning=melspectrogram,
                global_conditioning=global_conditioning,
            )

        return waveform, melspectrogram, log_est_duration

    def remove_weight_norm_(self) -> None:
        unwrapped_feat_to_wave = unwrap(self.feat_to_wave)

        if hasattr(unwrapped_feat_to_wave, "remove_weight_norm_"):
            unwrapped_feat_to_wave.remove_weight_norm_()


class FastSpeechWaveNetBridge(nn.Module):
    def __init__(self, take_exp: bool = False) -> None:
        super().__init__()

        self.take_exp = take_exp = take_exp

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Transform output of FastSpeech to input of WaveNet.

        Args:
            input (torch.Tensor): Melspectrogram of shape (batch_size, length, n_mels)
                in log.

        Returns:
            torch.Tensor: Melspectrogram of shape (batch_size, n_mels, length).
        """
        x = input.permute(0, 2, 1)

        if self.take_exp:
            output = torch.exp(x)
        else:
            output = x

        return output
