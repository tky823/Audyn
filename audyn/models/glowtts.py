from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Independent
from torch.distributions.normal import Normal

from ..modules.flow import BaseFlow
from ..modules.glowtts import (
    MaskedActNorm1d,
    MaskedInvertiblePointwiseConv1d,
    MaskedWaveNetAffineCoupling,
)

__all__ = ["TextEncoder", "Decoder"]


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
