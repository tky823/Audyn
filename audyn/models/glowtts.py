import math
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal

from ..modules.fastspeech import FFTrBlock
from ..modules.flow import BaseFlow
from ..modules.glowtts import (
    MaskedActNorm1d,
    MaskedInvertiblePointwiseConv1d,
    MaskedWaveNetAffineCoupling,
)
from ..utils.alignment.monotonic_align import search_monotonic_alignment_by_viterbi
from ..utils.duration import transform_log_duration
from .fastspeech import _get_clones

__all__ = [
    "GlowTTS",
    "TextEncoder",
    "Encoder",
    "Decoder",
    "GlowTTSTransformerEncoder",
    "TransformerEncoder",
]


class GlowTTS(nn.Module):
    """GlowTTS proposed by https://arxiv.org/abs/2005.11129.

    Args:
        encoder (nn.Module): Module to transform text tokens into latent features. The module
            takes positional argument of text tokens (batch_size, max_src_length) and keyword
            argument ``padding_mask`` (batch_size, src_length).
        decoder (audyn.modules.flow.BaseFlow): Flow to transform spectrograms into latent features
            in forward pass. The module takes positional argument of target features
            (batch_size, max_tgt_length, num_tgt_features) and keyword arguments ``padding_mask``
            (batch_size, max_tgt_length), ``logdet`` (batch_size,), and ``reverse`` (bool).
        duration_predictor (nn.Module): Duration predictor which takes positional argument of
            text features (batch_size, max_src_length, num_src_features) and keyword argument
            ``padding_mask`` (batch_size, max_src_length).

    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: BaseFlow,
        duration_predictor: nn.Module,
        length_regulator: nn.Module,
        transform_middle: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.length_regulator = length_regulator
        self.transform_middle = transform_middle
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_length: Optional[torch.LongTensor] = None,
        tgt_length: Optional[torch.LongTensor] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.LongTensor,
    ]:
        """Forward pass of GlowTTS.

        .. note::

            Only ``batch_first=True`` is supported.

        Args:
            src (torch.Tensor): Source feature of shape (batch_size, max_src_length).
            tgt (torch.Tensor): Target feature of shape (batch_size, in_channels, max_tgt_length).
            src_length (torch.LongTensor): Source feature lengths of shape (batch_size,).
            tgt_length (torch.LongTensor): Target feature lengths of shape (batch_size,).

        Returns:
            tuple: Tuple of tensors containing:

                - tuple: Source latent variables (batch_size, max_src_length, num_features)
                    and target latent variables (batch_size, max_tgt_length, num_features).
                - tuple: Durations estimated by duration predictor in log-domain
                    and extracted by monotonic alignment search in linear-domain.
                    The shape is (batch_size, max_src_length).
                - tuple: Padding masks for source (batch_size, max_src_length)
                    and target (batch_size, max_tgt_length).
                - torch.Tensor: Log-determinant.
                - torch.LongTensor: Number of elements in a sample.

        """
        # Log-determinant is required for forward pass.
        logdet = 0

        if src_length is None:
            src_padding_mask = None
        else:
            max_src_length = src.size(-1)
            src_padding_mask = torch.arange(
                max_src_length, device=src.device
            ) >= src_length.unsqueeze(dim=-1)

        if tgt_length is None:
            tgt_padding_mask = None
        else:
            max_tgt_length = tgt.size(-1)
            tgt_padding_mask = torch.arange(
                max_tgt_length, device=tgt.device
            ) >= tgt_length.unsqueeze(dim=-1)

        src_latent, normal = self.encoder(src, padding_mask=src_padding_mask)

        # NOTE: "est_duration" might be taken log.
        log_est_duration = self.duration_predictor(src_latent, padding_mask=src_padding_mask)

        if src_padding_mask is not None:
            log_est_duration = log_est_duration.masked_fill(src_padding_mask, -float("inf"))

        tgt_latent = self.decoder(
            tgt,
            padding_mask=tgt_padding_mask,
            logdet=logdet,
            reverse=False,
        )
        if tgt_padding_mask is None:
            tgt_latent, z_logdet = tgt_latent
        else:
            tgt_latent, tgt_padding_mask, z_logdet = tgt_latent

            if tgt_padding_mask.dim() == 3:
                tgt_padding_mask = torch.sum(tgt_padding_mask, dim=1)
                tgt_padding_mask = tgt_padding_mask.bool()
            elif tgt_padding_mask.dim() != 2:
                raise ValueError(
                    "tgt_padding_mask should be 2 or 3D, ",
                    f"but {tgt_padding_mask.dim()}D is found.",
                )

        mas_padding_mask = src_padding_mask.unsqueeze(dim=-2) | tgt_padding_mask.unsqueeze(dim=-1)

        if self.transform_middle is not None:
            tgt_latent = self.transform_middle(tgt_latent)

        log_prob_z, ml_duration = self.search_gaussian_monotonic_alignment(
            tgt_latent,
            normal,
            padding_mask=mas_padding_mask,
        )
        logdet = log_prob_z.sum(dim=-1) + z_logdet

        latent = src_latent, tgt_latent
        duration = log_est_duration, ml_duration
        padding_mask = src_padding_mask, tgt_padding_mask

        return latent, duration, padding_mask, logdet

    @torch.no_grad()
    def inference(
        self,
        src: torch.Tensor,
        src_length: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
        noise_scale: float = 1,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Inference of GlowTTS.

        Args:
            src (torch.Tensor): Source feature of shape (batch_size, max_src_length).
            src_length (torch.LongTensor): Source feature lengths of shape (batch_size,).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length * src_length.
            noise_scale (float): Parameter to scale noise. Default: ``1``.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Log-likelihood of target feature of shape (batch_size,).
                - torch.LongTensor: Estimated duration in linear-domain
                    of shape (batch_size, max_src_length).

        .. note::

            Sampling is performed by ``torch.distributions.distribution.Distribution.sample``.

        """
        if src_length is None:
            src_padding_mask = None
        else:
            max_src_length = torch.max(src_length).item()
            src_padding_mask = torch.arange(
                max_src_length, device=src.device
            ) >= src_length.unsqueeze(dim=-1)

        src_latent, normal = self.encoder(src, padding_mask=src_padding_mask)

        # NOTE: "est_duration" might be taken log.
        log_est_duration = self.duration_predictor(src_latent, padding_mask=src_padding_mask)

        # NOTE: transform_log_duration may apply flooring.
        linear_est_duration = transform_log_duration(log_est_duration)

        if src_padding_mask is not None:
            linear_est_duration = linear_est_duration.masked_fill(src_padding_mask, 0)

        if isinstance(normal, tuple):
            mean, log_std = normal
            stddev = torch.exp(log_std)
        elif isinstance(normal, (Normal, Independent)):
            mean, stddev = normal.mean, normal.stddev
        else:
            raise ValueError(f"{type(normal)} is not supported.")

        stddev = noise_scale * stddev
        expanded_mean, scaled_linear_est_duration = self.length_regulator(
            mean,
            linear_est_duration,
            padding_mask=src_padding_mask,
            max_length=max_length,
        )
        expanded_stddev, _ = self.length_regulator(
            stddev,
            linear_est_duration,
            padding_mask=src_padding_mask,
            max_length=max_length,
        )

        tgt_length = scaled_linear_est_duration.sum(dim=1)

        # To avoid error of stddev=0 in Normal class,
        # we use standard Gaussian distribution here.
        zeros = torch.zeros_like(expanded_mean)
        ones = torch.ones_like(expanded_stddev)
        normal = Normal(loc=zeros, scale=ones)
        normal = Independent(normal, reinterpreted_batch_ndims=1)
        tgt_latent = expanded_mean + expanded_stddev * normal.sample()

        tgt_length = scaled_linear_est_duration.sum(dim=1)
        max_tgt_length = torch.max(tgt_length).item()
        tgt_padding_mask = self.create_length_padding_mask(tgt_length, max_length=max_tgt_length)

        if self.transform_middle is not None:
            tgt_latent = self.transform_middle(tgt_latent)

        # tgt_latent: Latent variables of (batch_size, max_length, num_features),
        #    where num_features is typically number of bins in MelSpectrogram.
        if hasattr(self.decoder, "inference") and callable(self.decoder.inference):
            output, tgt_padding_mask = self.decoder.inference(
                tgt_latent,
                padding_mask=tgt_padding_mask,
            )
        else:
            output, tgt_padding_mask = self.decoder(
                tgt_latent,
                padding_mask=tgt_padding_mask,
                reverse=True,
            )

        return output, linear_est_duration

    @staticmethod
    def search_gaussian_monotonic_alignment(
        input: torch.Tensor,
        normal: Union[Independent, Tuple[torch.Tensor, torch.Tensor]],
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Search monotonic alignment assuming Gaussian distributions.

        Args:
            input (torch.Tensor): Probablistic variable
                of shape (batch_size, max_tgt_length, num_features).
            normal (torch.distributions.Independent or tuple): Gaussian distribution parametrized
                by mean (batch_size, max_src_length, num_features) and
                stddev (batch_size, max_src_length, num_features).
            padding_mask (torch.BoolTensor, optional): Padding mask for monotonic alignment.

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Maximum log-likelihood of p(z) of
                    shape (batch_size, max_tgt_length).
                - torch.LongTensor: Duration in linear-domain of
                    shape (batch_size, max_src_length).

        """
        batch_size, max_tgt_length, num_features = input.size()

        if isinstance(normal, tuple):
            mean, log_std = normal
            _, max_src_length, _ = mean.size()
            x = input.unsqueeze(dim=-2)
            mean = mean.unsqueeze(dim=-3)
            log_std = log_std.unsqueeze(dim=-3)
            x = (x - mean) / torch.exp(log_std)
            log_prob = -0.5 * torch.sum(x**2, dim=-1)
            log_prob = log_prob - 0.5 * num_features * math.log(2 * math.pi) - log_std.sum(dim=-1)
        elif isinstance(normal, Independent):
            _, max_src_length, _ = normal.mean.size()
            x = input.permute(1, 0, 2).contiguous()
            x = x.unsqueeze(dim=2)
            log_prob = normal.log_prob(x)
            log_prob = log_prob.permute(1, 0, 2).contiguous()
        else:
            raise ValueError(f"{type(normal)} is not supported.")

        assert log_prob.size() == (batch_size, max_tgt_length, max_src_length)

        hard_alignment = search_monotonic_alignment_by_viterbi(
            log_prob,
            padding_mask=padding_mask,
            take_log=False,
        )
        # sum along src_length dimension
        log_prob = torch.sum(log_prob * hard_alignment, dim=-1)
        duration = hard_alignment.sum(dim=1)

        return log_prob, duration

    @staticmethod
    def create_length_padding_mask(
        length: torch.Tensor, max_length: Optional[int] = None
    ) -> torch.BoolTensor:
        """Create padding mask for length tensors.

        Args:
            length (torch.Tensor): Lengths of shape (batch_size,)
            max_length (int, optional): Max value of lengths.

        Returns:
            torch.BoolTensor of padding mask.
            The shape is (batch_size, max_length)

        """
        if max_length is None:
            max_length = torch.max(length).item()

        # Allocation is required for mps
        indices = torch.arange(max_length).to(length.device)
        padding_mask = indices >= length.unsqueeze(dim=-1)

        return padding_mask


class TextEncoder(nn.Module):
    """Text Encoder of GlowTTS."""

    def __init__(
        self,
        word_embedding: nn.Module,
        backbone: nn.Module,
        proj_mean: nn.Module,
        proj_std: nn.Module,
        pre_net: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.word_embedding = word_embedding
        self.pre_net = pre_net
        self.backbone = backbone
        self.proj_mean = proj_mean
        self.proj_std = proj_std

    def forward(
        self,
        input: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> Tuple[torch.Tensor, Independent]:
        """Forward pass of TextEncoder.

        Args:
            input (torch.LongTensor): Text input of shape (batch_size, max_length).
            padding_mask (torch.Tensor): Padding mask of shape (batch_size, max_length).

        Returns:

            tuple: Tuple of tensors containing

                - torch.Tensor: Latent feature of shape (batch_size, max_length, out_channels),
                    where ``out_channels`` is determined by ``self.backbone``.
                - torch.distributions.Independent: Multivariate Gaussian distribution
                    composed by ``mean`` and ``stddev``.

        """
        x = self.word_embedding(input)
        x = self._apply_mask(x, padding_mask=padding_mask)

        if self.pre_net is not None:
            x = self.pre_net(x, padding_mask=padding_mask)
            x = self._apply_mask(x, padding_mask=padding_mask)

        output = self.backbone(x, src_key_padding_mask=padding_mask)
        output = self._apply_mask(output, padding_mask=padding_mask)

        mean = self.proj_mean(output)
        log_std = self.proj_std(output)
        mean = self._apply_mask(mean, padding_mask=padding_mask)
        log_std = self._apply_mask(log_std, padding_mask=padding_mask)

        normal = Normal(loc=mean, scale=torch.exp(log_std))
        normal = Independent(normal, reinterpreted_batch_ndims=1)

        return output, normal

    @staticmethod
    def _apply_mask(
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if padding_mask is not None:
            output = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
        else:
            output = input

        return output


class Encoder(TextEncoder):
    """Wrapper class of TextEncoder."""


class Decoder(BaseFlow):
    """Decoder of GlowTTS."""

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_flows: int = 12,
        num_layers: int = 4,
        num_splits: int = 4,
        down_scale: int = 2,
        kernel_size: int = 5,
        dilation_rate: int = 5,
        bias: bool = True,
        is_causal: bool = False,
        conv: str = "gated",
        dropout: float = 0,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        scaling_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if isinstance(in_channels, list):
            raise NotImplementedError("List is not supported as in_channels now.")

        self.in_channels = in_channels
        self.num_flows = num_flows
        self.down_scale = down_scale

        backbone = []

        for _ in range(num_flows):
            backbone.append(
                GlowBlock(
                    in_channels * down_scale,
                    hidden_channels,
                    skip_channels=skip_channels,
                    num_layers=num_layers,
                    num_splits=num_splits,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    bias=bias,
                    is_causal=is_causal,
                    conv=conv,
                    dropout=dropout,
                    weight_norm=weight_norm,
                    split=split,
                    concat=concat,
                    scaling=scaling,
                    scaling_channels=scaling_channels,
                )
            )

        self.backbone = nn.ModuleList(backbone)

    def _forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of GlowTTS Decoder.

        Args:
            input (torch.Tensor): 3D tensor of shape
                (batch_size, in_channels * down_scale, length // down_scale).

        Returns:
            tuple of torch.Tensor.

        """
        num_flows = self.num_flows
        return_logdet = logdet is not None

        x = input

        for flow_idx in range(num_flows):
            x = self.backbone[flow_idx](
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=False,
            )
            if return_logdet:
                x, logdet = x

        output = x

        if return_logdet:
            return output, logdet
        else:
            return output

    def _inference(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: torch.Tensor = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Reverse pass of GlowTTS Decoder.

        Args:
            input (torch.Tensor): 3D tensor of shape
                (batch_size, in_channels * down_scale, length // down_scale).

        Returns:
            tuple of torch.Tensor.

        """
        num_flows = self.num_flows
        return_logdet = logdet is not None

        x = input

        for flow_idx in range(num_flows - 1, -1, -1):
            x = self.backbone[flow_idx](
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=True,
            )
            if return_logdet:
                x, logdet = x

        output = x

        if return_logdet:
            return output, logdet
        else:
            return output

    @staticmethod
    def _expand_padding_mask(
        padding_mask: torch.BoolTensor,
        other: torch.Tensor,
    ) -> torch.BoolTensor:
        """Expand padding mask.

        Args:
            padding_mask (torch.BoolTensor): Padding mask of shape
                (batch_size, length) or (batch_size, num_features, length).
            other (torch.Tensor): Tensor of shape (batch_size, num_features, length).

        Returns:
            torch.BoolTensor: Expanded padding mask of shape (batch_size, num_features, length).

        """
        if padding_mask.dim() == 2:
            padding_mask = padding_mask.unsqueeze(dim=1)
        elif padding_mask.dim() != 3:
            raise ValueError(f"{padding_mask.dim()}D mask is not supported.")

        expanded_padding_mask = padding_mask.expand(other.size())

        return expanded_padding_mask

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Forward pass of GlowTTS Decoder.

        Args:
            input (torch.Tensor): Acoustic feature or latent variable of shape
                (batch_size, in_channels, length).
            padding_mask (torch.Tensor, optional): Padding mask of shape (batch_size, length).
                or (batch_size, in_channels, length).
            logdet (torch.Tensor, optional): Log-determinant of shape (batch_size,).

        Returns:
            torch.Tensor or tuple: Tensor or tuple of tensors.

                - If ``padding_mask`` is ``None`` and ``logdet`` is None,
                  output (batch_size, in_channels, length) is returned.
                - If ``padding_mask`` is given and ``logdet`` is None,
                  output and modified padding mask (batch_size, in_channels, length) are returned.
                - If ``padding_mask`` is ``None`` and ``logdet`` is given,
                  output and log-determinant (batch_size,) are returned.
                - If ``padding_mask`` and ``logdet`` are given, output, modified padding mask,
                    and log-determinant are returned.

        """
        down_scale = self.down_scale

        length = input.size(-1)
        padding = (down_scale - length % down_scale) % down_scale
        x = F.pad(input, (0, padding))
        x = self.squeeze(x, down_scale=down_scale)

        if padding_mask is None:
            padding_mask_dim = None
            expanded_padding_mask = None
        else:
            padding_mask_dim = padding_mask.dim()
            expanded_padding_mask = self._expand_padding_mask(padding_mask, input)
            expanded_padding_mask = F.pad(expanded_padding_mask, (0, padding), value=True)
            expanded_padding_mask = self.squeeze(expanded_padding_mask, down_scale=down_scale)

            # overwrite padding mask to round down sequence length
            padding_mask = torch.sum(expanded_padding_mask, dim=1)
            padding_mask = padding_mask.bool()
            expanded_padding_mask = self._expand_padding_mask(padding_mask, x)
            x = x.masked_fill(expanded_padding_mask, 0)

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)
        return_logdet = logdet is not None

        if reverse:
            x = self._inference(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
            )
        else:
            x = self._forward(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
            )

        if return_logdet:
            x, logdet = x

        x = self.unsqueeze(x, up_scale=down_scale)
        output = F.pad(x, (0, -padding))

        if padding_mask is not None:
            # i.e. expanded_padding_mask is not None
            padding_mask = self.unsqueeze(expanded_padding_mask, up_scale=down_scale)
            padding_mask = F.pad(padding_mask, (0, -padding))

            if padding_mask_dim == 2:
                # 3D to 2D
                padding_mask = torch.sum(padding_mask, dim=1)
                padding_mask = padding_mask.bool()

        if return_logdet:
            if padding_mask is None:
                return output, logdet
            else:
                return output, padding_mask, logdet
        else:
            if padding_mask is None:
                return output
            else:
                return output, padding_mask

    @torch.no_grad()
    def inference(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: Optional[torch.Tensor] = None,
    ) -> Union[Any, Tuple[Any, torch.Tensor]]:
        """Inference of Decoder in GlowTTS.

        Args:
            input (torch.Tensor): Latent variable of shape (batch_size, in_channels, length).
            padding_mask (torch.Tensor, optional): Padding mask of shape (batch_size, length).
            logdet (torch.Tensor, optional): Log-determinant of shape (batch_size,).

        Returns:
            tuple: Tuple of tensors.

        """
        return self.forward(
            input,
            padding_mask=padding_mask,
            logdet=logdet,
            reverse=True,
        )

    @staticmethod
    def squeeze(
        input: Union[torch.Tensor, torch.BoolTensor],
        down_scale: int,
    ) -> Union[torch.Tensor, torch.BoolTensor]:
        """Squeeze tensor.

        Args:
            input (torch.Tensor or torch.BoolTensor): 3D tensor of shape
                (batch_size, num_features, length). ``length`` should be divisible
                by ``down_scale``.
            down_scale (int): Down scale of time axis.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch_size, down_scale * in_channels, length // down_scale).

        """
        batch_size, in_channels, length = input.size()

        x = input.view(batch_size, in_channels, length // down_scale, down_scale)
        x = x.permute(0, 3, 1, 2).contiguous()
        output = x.view(batch_size, down_scale * in_channels, length // down_scale)

        return output

    @staticmethod
    def unsqueeze(
        input: Union[torch.Tensor, torch.BoolTensor],
        up_scale: int,
    ) -> Union[torch.Tensor, torch.BoolTensor]:
        """Unsqueeze tensor.

        Args:
            input (torch.Tensor or torch.BoolTensor): 3D tensor of shape
                (batch_size, num_features, length). ``length`` should be divisible
                by ``up_scale``.
            up_scale (int): Up scale of time axis.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch_size, in_channels // up_scale, length * up_scale).

        """
        batch_size, in_channels, length = input.size()

        output = input.view(batch_size, up_scale, in_channels // up_scale, length)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, in_channels // up_scale, length * up_scale)

        return output


class PreNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int = 5,
        dropout: float = 0.5,
        batch_first: bool = True,
        num_layers: int = 3,
    ) -> None:
        super().__init__()

        self.num_layers = num_layers

        backbone = []

        for layer_idx in range(num_layers):
            if layer_idx == 0:
                _in_channels = in_channels
            else:
                _in_channels = hidden_channels

            _out_channels = hidden_channels
            conv1d = ConvBlock(
                _in_channels,
                _out_channels,
                kernel_size=kernel_size,
                dropout=dropout,
                batch_first=batch_first,
            )
            backbone.append(conv1d)

        self.backbone = nn.ModuleList(backbone)

        _in_channels = hidden_channels
        _out_channels = out_channels

        self.proj = nn.Linear(_in_channels, _out_channels)

        self._reset_parameters()

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = input

        for layer in self.backbone:
            if padding_mask is None:
                x = layer(x)
            else:
                x = layer(x, padding_mask=padding_mask)

        output = self.proj(x)

        return output

    def _reset_parameters(self) -> None:
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dropout: float = 0.5,
        batch_first: bool = True,
    ) -> None:
        super().__init__()

        assert kernel_size % 2 == 1, "kernel_size should be odd."

        self.kernel_size = kernel_size
        self.batch_first = batch_first

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1)
        self.norm1d = nn.LayerNorm(out_channels)
        self.nonlinear1d = nn.ReLU()
        self.dropout1d = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        kernel_size = self.kernel_size
        batch_first = self.batch_first

        if batch_first:
            x = input.permute(0, 2, 1).contiguous()
        else:
            x = input.permute(1, 2, 0).contiguous()

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(dim=-2), 0)

        x = F.pad(x, (kernel_size // 2, kernel_size // 2))
        x = self.conv1d(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.norm1d(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

        if not batch_first:
            x = x.permute(1, 0, 2).contiguous()

        x = self.nonlinear1d(x)
        output = self.dropout1d(x)

        return output


class GlowTTSTransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        batch_first: bool = False,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

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
        r"""Forward pass of feed-forward transformer block for GlowTTS.

        Args:
            src (torch.LongTensor): Source feature of shape (batch_size, src_length, embed_dim)
                or (src_length, batch_size, embed_dim).
            mask (torch.Tensor): Attention mask for source of shape
                (src_length, src_length) or (batch_size * num_heads, src_length, src_length).
            src_key_padding_mask (torch.Tensor): Padding mask of shape (src_length,)
                or (batch_size, src_length).

        Returns:
            torch.Tensor: Encoded feature of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise, (src_length, batch_size, embed_dim).

        """
        x = src

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


class TransformerEncoder(GlowTTSTransformerEncoder):
    """Wrapper class of GlowTTSTransformerEncoder."""


class GlowBlock(BaseFlow):
    """Glow block for GlowTTS.

    Args:
        scaling (bool): Whether to use scaling factor. Default: ``False``.
        scaling_channels (int, optional): Number of channels of scaling factor.
            If ``scaling=True``, this parameter is activated. In that case,
            ``scaling_channels`` is expected ``None`` or ``coupling_channels``.
            ``coupling_channels`` is typically ``in_channels[0]`` or ``in_channels // 2``.

    """

    # TODO: improve docs

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 4,
        num_splits: int = 4,
        kernel_size: int = 5,
        dilation_rate: int = 1,
        bias: bool = True,
        is_causal: bool = False,
        conv: str = "gated",
        dropout: float = 0,
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        scaling_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        # TODO: support list type input properly
        if type(in_channels) is list:
            num_features = sum(in_channels)
            coupling_channels = in_channels[0]
        else:
            num_features = in_channels
            coupling_channels = in_channels // 2

        self.norm1d = MaskedActNorm1d(num_features)
        self.conv1d = MaskedInvertiblePointwiseConv1d(num_splits)
        self.affine_coupling = MaskedWaveNetAffineCoupling(
            coupling_channels,
            hidden_channels,
            skip_channels=skip_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            bias=bias,
            is_causal=is_causal,
            conv=conv,
            dropout=dropout,
            weight_norm=weight_norm,
            split=split,
            concat=concat,
            scaling=scaling,
            scaling_channels=scaling_channels,
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert input.dim() == 3, "input is expected to be 3D tensor, but given {}D tensor.".format(
            input.dim()
        )

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)
        return_logdet = logdet is not None

        if reverse:
            x = self.affine_coupling(
                input,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            x = self.conv1d(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            output = self.norm1d(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )
        else:
            x = self.norm1d(
                input,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            x = self.conv1d(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            output = self.affine_coupling(
                x,
                logdet=logdet,
                reverse=reverse,
            )

        if return_logdet:
            output, logdet = output

            return output, logdet
        else:
            return output
