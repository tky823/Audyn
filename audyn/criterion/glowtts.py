from typing import Optional

import torch

from .flow import NLLLoss


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
