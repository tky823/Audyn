from typing import Dict, Optional, Union

import torch

from .fastspeech import FastSpeechMSELoss
from .flow import NLLLoss

__all__ = ["GlowTTSNLLLoss", "GlowTTSDurationLoss"]


class GlowTTSNLLLoss(NLLLoss):
    """NLLLoss for GlowTTS."""

    def forward(
        self,
        logdet: torch.Tensor,
        tgt: torch.Tensor,
        tgt_padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of general flow loss.

        This loss just reverse the sign of input log-determinant.

        Args:
            logdet (torch.Tensor): Log-determinant of shape (batch_size,).
            tgt (torch.Tensor): Target feature of shape (batch_size, n_mels, length).
            tgt_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (batch_size, length).

        Returns:
            torch.Tensor: Negative log-likelihood.

        """
        if tgt_padding_mask.dim() == 2:
            tgt_padding_mask = tgt_padding_mask.unsqueeze(dim=-2)
        elif tgt_padding_mask.dim() != 3:
            raise ValueError("tgt_padding_mask should be 2D or 3D.")

        tgt_padding_mask = tgt_padding_mask.expand(tgt.size())
        tgt_non_padding_mask = torch.logical_not(tgt_padding_mask)
        num_elements = tgt_non_padding_mask.sum(dim=(1, 2))

        loss = super().forward(logdet / num_elements)

        return loss


class GlowTTSDurationLoss(FastSpeechMSELoss):
    def __init__(
        self,
        take_log: Union[bool, Dict[str, bool]] = False,
        reduction: Optional[str] = None,
        batch_first: bool = False,
        min: Optional[float] = None,
        max: Optional[float] = None,
    ) -> None:
        super().__init__(
            take_log=take_log,
            reduction=reduction,
            batch_first=batch_first,
            min=min,
            max=max,
        )

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass of duration MSE loss used in paper of GlowTTS.

        Args:
            input (torch.Tensor): Estimated feature of shape (batch_size, length)
                if ``batch_first=True``, otherwise (length, batch_size).
            target (torch.Tensor): Target feature of shape (batch_size, length)
                if ``batch_first=True``, otherwise (length, batch_size).

        Returns:
            torch.Tensor: Mean squared error. If ``reduction=None``, shape is same as input.
                If ``reduction=mean``, shape is ().

        .. note::

            Unlike duration loss used in FastSpeech, maximum duration of target
            feature (e.g. Melspectrogram) might be modified by GlowTTS decoder.
            The modified duration is judged by ``torch.count_nonzero`` of ``target``.

        """
        if self.batch_first:
            length = torch.count_nonzero(target, dim=-1)
        else:
            length = torch.count_nonzero(target, dim=0)

        loss = super().forward(input, target, length=length)

        return loss
