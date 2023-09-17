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
from ..utils.alignment.monotonic_align import viterbi_monotonic_alignment
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
    def __init__(
        self,
        encoder: nn.Module,
        decoder: BaseFlow,
        duration_predictor: nn.Module,
        length_regulator: nn.Module,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.duration_predictor = duration_predictor
        self.length_regulator = length_regulator
        self.decoder = decoder

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_length: Optional[torch.LongTensor] = None,
        tgt_length: Optional[torch.LongTensor] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of GlowTTS.

        Args:
            src (torch.Tensor): Source feature of shape (batch_size, src_length).
            tgt (torch.Tensor): Target feature of shape (batch_size, in_channels, src_length).
            src_length (torch.LongTensor): Source feature lengths of shape (batch_size,).
            tgt_length (torch.LongTensor): Target feature lengths of shape (batch_size,).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length * src_length.

        Returns:
            tuple: Tuple of tensors containing:

                - tuple: Source latent variables (batch_size, src_length, num_features)
                    and target latent variables (batch_size, tgt_length, num_features).
                - tuple: Durations estimated by duration predictor in log-domain
                    and extracted by monotonic alignment search in linear-domain.
                    The shape is (batch_size, src_length).
                - tuple: Padding masks for source (batch_size, src_length)
                    and target (batch_size, tgt_length).
                - torch.Tensor: Log-determinant.

        """
        # Log-determinant is required for forward pass.
        logdet = 0

        if src_length is None:
            src_padding_mask = None
        else:
            max_src_length = torch.max(src_length).item()
            src_padding_mask = torch.arange(
                max_src_length, device=src.device
            ) >= src_length.unsqueeze(dim=-1)

        if tgt_length is None:
            tgt_padding_mask = None
        else:
            max_tgt_length = torch.max(tgt_length).item()
            tgt_padding_mask = torch.arange(
                max_tgt_length, device=tgt.device
            ) >= tgt_length.unsqueeze(dim=-1)

        src_latent, normal = self.encoder(src, padding_mask=src_padding_mask)

        if isinstance(normal, tuple):
            mean, log_std = normal
            normal = Normal(loc=mean, scale=torch.exp(log_std))
            normal = Independent(normal, reinterpreted_batch_ndims=1)
        elif isinstance(normal, Normal):
            normal = Independent(normal, reinterpreted_batch_ndims=1)
        elif not isinstance(normal, Independent):
            raise ValueError(f"{type(normal)} is not supported.")

        # NOTE: "est_duration" might be taken log.
        log_est_duration = self.duration_predictor(src_latent)
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

        # TODO: explicit permutation
        tgt_latent = tgt_latent.permute(0, 2, 1).contiguous()
        log_prob_z, ml_duration = self.search_gaussian_monotonic_alignment(tgt_latent, normal)
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
            src (torch.Tensor): Source feature of shape (batch_size, src_length).
            src_length (torch.LongTensor): Source feature lengths of shape (batch_size,).
            max_length (int, optional): Maximum length of source duration.
                The output length is up to max_length * src_length.
            noise_scale (float): Parameter to scale noise. Default: ``1``.

        Returns:
            tuple: Tuple of tensors containing:

                - torch.Tensor: Log-likelihood of target feature of shape (batch_size,).
                - torch.LongTensor: Estimated duration in linear-domain
                    of shape (batch_size, src_length).

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
        log_est_duration = self.duration_predictor(src_latent)
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

        # TODO: explicit permutation
        tgt_latent = tgt_latent.permute(0, 2, 1).contiguous()

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
        input: torch.Tensor, normal: Independent
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """Search monotonic alignment assuming Gaussian distributions.

        Args:
            input (torch.Tensor): Probablistic variable
                of shape (batch_size, tgt_length, num_features).
            normal (torch.distributions.Independent): Gaussian distribution parametrized by
                mean (batch_size, src_length, num_features) and
                stddev (batch_size, src_length, num_features),

        Returns:
            tuple: Tuple of tensors containing

                - torch.Tensor: Maximum log-likelihood of p(z) of
                    shape (batch_size, tgt_length, src_length).
                - torch.LongTensor: Duration in linear-domain of
                    shape (batch_size, src_length).

        """
        batch_size, tgt_length, _ = input.size()
        _, src_length, _ = normal.mean.size()

        x = input.permute(1, 0, 2).contiguous()
        x = x.unsqueeze(dim=2)
        log_prob = normal.log_prob(x)
        log_prob = log_prob.permute(1, 0, 2).contiguous()

        assert log_prob.size() == (batch_size, tgt_length, src_length)

        hard_alignment = viterbi_monotonic_alignment(log_prob, take_log=False)
        log_prob = torch.sum(log_prob * hard_alignment, dim=1)
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
    ) -> None:
        super().__init__()

        self.word_embedding = word_embedding
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
            input (torch.LongTensor): Text input of shape (batch_size, length).
            padding_mask (torch.Tensor): Padding mask of shape (batch_size, length).

        Returns:

            tuple: Tuple of tensors containing

                - torch.Tensor: Latent feature of shape (batch_size, length, out_channels),
                    where ``out_channels`` is determined by ``self.backbone``.
                - torch.distributions.Independent: Multivariate Gaussian distribution
                    composed by ``mean`` and ``stddev``.

        """
        x = self.word_embedding(input)
        x = self._apply_mask(x, padding_mask=padding_mask)
        output = self.backbone(x)
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
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
    ) -> None:
        super().__init__()

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
                    bias=bias,
                    causal=causal,
                    conv=conv,
                    weight_norm=weight_norm,
                    split=split,
                    concat=concat,
                    scaling=scaling,
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
            tuple: Tuple of tensors.

        """
        down_scale = self.down_scale

        length = input.size(-1)
        padding = (down_scale - length % down_scale) % down_scale
        x = F.pad(input, (0, padding))
        x = self.squeeze(x)

        if padding_mask is None:
            padding_mask_dim = None
            expanded_padding_mask = None
        else:
            padding_mask_dim = padding_mask.dim()
            expanded_padding_mask = self._expand_padding_mask(padding_mask, input)
            expanded_padding_mask = F.pad(expanded_padding_mask, (0, padding), value=True)
            expanded_padding_mask = self.squeeze(expanded_padding_mask)

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

        x = self.unsqueeze(x)
        output = F.pad(x, (0, -padding))

        if padding_mask is not None:
            # i.e. expanded_padding_mask is not None
            padding_mask = self.unsqueeze(expanded_padding_mask)
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

    def squeeze(
        self,
        input: Union[torch.Tensor, torch.BoolTensor],
    ) -> Union[torch.Tensor, torch.BoolTensor]:
        """Squeeze tensor.

        Args:
            input (torch.Tensor or torch.BoolTensor): 3D tensor of shape
                (batch_size, num_features, length). ``length`` should be divisible
                by ``self.down_scale``.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch_size, down_scale * in_channels, length // down_scale).

        """
        down_scale = self.down_scale
        batch_size, in_channels, length = input.size()

        x = input.view(batch_size, in_channels, length // down_scale, down_scale)
        x = x.permute(0, 3, 1, 2).contiguous()
        output = x.view(batch_size, down_scale * in_channels, length // down_scale)

        return output

    def unsqueeze(
        self, input: Union[torch.Tensor, torch.BoolTensor]
    ) -> Union[torch.Tensor, torch.BoolTensor]:
        """Unsqueeze tensor.

        Args:
            input (torch.Tensor or torch.BoolTensor): 3D tensor of shape
                (batch_size, num_features, length). ``length`` should be divisible
                by ``self.down_scale``.

        Returns:
            torch.Tensor: Reshaped tensor of shape
                (batch_size, in_channels // down_scale, length * down_scale).

        """
        down_scale = self.down_scale
        batch_size, in_channels, length = input.size()

        output = input.view(batch_size, down_scale, in_channels // down_scale, length)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = output.view(batch_size, in_channels // down_scale, length * down_scale)

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
    def __init__(
        self,
        in_channels: Union[int, List[int]],
        hidden_channels: int,
        skip_channels: Optional[int] = None,
        num_layers: int = 4,
        num_splits: int = 4,
        kernel_size: int = 5,
        bias: bool = True,
        causal: bool = False,
        conv: str = "gated",
        weight_norm: bool = True,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
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
            bias=bias,
            causal=causal,
            conv=conv,
            weight_norm=weight_norm,
            split=split,
            concat=concat,
            scaling=scaling,
            in_channels=in_channels,
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
            x = self.conv1d(
                input,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            x = self.norm1d(
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
        else:
            x = self.affine_coupling(
                input,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            x = self.norm1d(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

            if return_logdet:
                x, logdet = x

            output = self.conv1d(
                x,
                padding_mask=padding_mask,
                logdet=logdet,
                reverse=reverse,
            )

        if return_logdet:
            output, logdet = output

            return output, logdet
        else:
            return output
