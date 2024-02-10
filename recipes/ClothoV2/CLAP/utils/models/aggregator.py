from typing import Optional, Union

import torch
import torch.nn as nn


class Aggregator(nn.Module):
    """Wrapper of transformer to aggregate sequence feature.

    Args:
        batch_first (bool): Whether to batch dimension is first or not.
        aggregation (str): Aggregation type.

    """

    def __init__(
        self,
        batch_first: bool = True,
        aggregation: str = "pool",
    ) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.aggregation = aggregation

    def forward(
        self,
        input: Union[torch.LongTensor, torch.Tensor],
        length: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Aggregate sequence feature.

        Args:
            input (torch.Tensor): Input sequence including class token of shape
                (batch_size, max_length + 1, embedding_dim) if ``batch_first=True``.
                Otherwise, (batch_size, embedding_dim, max_length + 1).
            length (torch.LongTensor, optional): Lengths of each sequence (batch_size,).

        Returns:
            torch.Tensor: Aggregated feature. If ``aggregation=cls``, class token of shape
                (embedding_dim,) is returned. If ``aggregation=pool``, pooled feature of shape
                (embedding_dim,) except for class token is returend. If ``aggregation=none``,
                input sequence including class token is returned.

        """
        aggregation = self.aggregation
        batch_first = self.batch_first

        factory_kwargs = {
            "dtype": torch.long,
            "device": input.device,
        }

        if batch_first:
            batch_size, max_length, _ = input.size()
        else:
            max_length, batch_size, _ = input.size()

        # NOTE: max_length includes cls token, so remove it.
        max_length = max_length - 1

        if length is None:
            length = torch.full((batch_size,), fill_value=max_length, **factory_kwargs)

        if batch_first:
            cls_token, output = torch.split(input, [1, max_length], dim=1)
        else:
            cls_token, output = torch.split(input, [1, max_length], dim=0)

        if aggregation == "cls":
            if batch_first:
                output = cls_token.squeeze(dim=1)
            else:
                output = cls_token.squeeze(dim=0)
        elif aggregation == "pool":
            if batch_first:
                output = output.sum(dim=1) / length.unsqueeze(dim=1)
            else:
                output = output.sum(dim=0) / length.unsqueeze(dim=0)
        elif aggregation == "none":
            output = input
        else:
            raise ValueError(f"{aggregation} is not supported as aggregation.")

        return output
