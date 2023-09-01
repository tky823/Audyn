import torch
import torch.nn as nn

__all__ = ["AbsolutePositionalEncoding"]


class AbsolutePositionalEncoding(nn.Module):
    def __init__(
        self,
        base: int = 10000,
        batch_first: bool = False,
    ) -> None:
        super().__init__()

        self.base = base
        self.batch_first = batch_first

    def forward(
        self,
        input: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of absolute positional encoding.

        Args:
            input (torch.Tensor): Input embeddings of shape (batch_size, length, num_features)
                if ``batch_first=True``, otherwise (length, batch_size, num_features).

        Returns:
            torch.Tensor: Output embeddings whose shape is same as input.

        """
        base = self.base
        batch_first = self.batch_first

        if batch_first:
            _, length, num_features = input.size()
        else:
            length, _, num_features = input.size()

        assert num_features % 2 == 0, "Feature dimension is expected to be even number."

        pos_seq = torch.arange(length)
        num_seq = torch.arange(0, num_features, 2) / num_features
        theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

        sin = torch.sin(theta)
        cos = torch.cos(theta)

        pos_emb = torch.stack([sin, cos], dim=2)  # (length, num_features // 2, 2)
        pos_emb = pos_emb.view(length, num_features)
        pos_emb = pos_emb.to(input.device)

        if not batch_first:
            pos_emb = pos_emb.unsqueeze(dim=1)

        output = input + pos_emb

        return output
