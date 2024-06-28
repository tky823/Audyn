import copy
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn as nn

from .tasnet import get_layer_norm

__all__ = [
    "DPRNN",
    "DPRNNBlock",
    "IntraChunkRNN",
    "InterChunkRNN",
]


class DPRNN(nn.Module):
    """Dual-path RNN."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        num_blocks: int = 6,
        is_causal: bool = False,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "lstm",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        backbone = []

        for _ in range(num_blocks):
            backbone.append(
                DPRNNBlock(
                    num_features,
                    hidden_channels,
                    norm=norm,
                    is_causal=is_causal,
                    rnn=rnn,
                    eps=eps,
                )
            )

        self.backbone = nn.Sequential(*backbone)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of dual-path RNN.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        output = self.backbone(input)

        return output


class DPRNNBlock(nn.Module):
    """Dual-path RNN block."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.intra_chunk_block = IntraChunkRNN(
            num_features, hidden_channels, norm=norm, rnn=rnn, eps=eps
        )
        self.inter_chunk_block = InterChunkRNN(
            num_features, hidden_channels, norm=norm, is_causal=is_causal, rnn=rnn, eps=eps
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of dual-path RNN block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        x = self.intra_chunk_block(input)
        output = self.inter_chunk_block(x)

        return output


class IntraChunkRNN(nn.Module):
    """Intra-chunk dual-path RNN."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_channels = hidden_channels

        num_directions = 2

        self.rnn = get_rnn(
            rnn,
            input_size=num_features,
            hidden_size=hidden_channels,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = get_layer_norm(norm, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm = "gLN"
                self.norm = get_layer_norm(norm, num_features, is_causal=False, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of intra-chunk dual-path RNN block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        num_features = self.num_features
        batch_size, _, inter_length, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input
        x = input.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size * inter_length, chunk_size, num_features)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.view(batch_size, inter_length * chunk_size, num_features)
        x = x.permute(0, 2, 1).contiguous()

        if self.norm is not None:
            x = self.norm(x)

        x = x.view(batch_size, num_features, inter_length, chunk_size)
        output = x + residual

        return output


class InterChunkRNN(nn.Module):
    """Inter-chunk dual-path RNN."""

    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        is_causal: bool = False,
        norm: Optional[Union[bool, str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = True,
        rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]] = "lstm",
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        self.num_features, self.hidden_channels = num_features, hidden_channels

        if is_causal:
            # uni-direction
            num_directions = 1
            self.rnn = get_rnn(
                rnn,
                input_size=num_features,
                hidden_size=hidden_channels,
                batch_first=True,
                bidirectional=False,
            )
        else:
            # bi-direction
            num_directions = 2
            self.rnn = get_rnn(
                rnn,
                input_size=num_features,
                hidden_size=hidden_channels,
                batch_first=True,
                bidirectional=True,
            )

        self.fc = nn.Linear(num_directions * hidden_channels, num_features)

        if norm is None:
            self.norm = None
        elif isinstance(norm, str):
            self.norm = get_layer_norm(norm, num_features, eps=eps)
        elif isinstance(norm, bool):
            if norm:
                norm = "cLN" if is_causal else "gLN"
                self.norm = get_layer_norm(norm, num_features, is_causal=is_causal, eps=eps)
            else:
                self.norm = None
        else:
            raise ValueError(f"{type(norm)} is not supported as norm.")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass of inter-chunk dual-path RNN block.

        Args:
            input (torch.Tensor): Input feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        Returns:
            torch.Tensor: Output feature of shape
                (batch_size, num_features, inter_length, chunk_size).

        """
        num_features = self.num_features
        batch_size, _, inter_length, chunk_size = input.size()

        self.rnn.flatten_parameters()

        residual = input
        x = input.permute(0, 3, 2, 1).contiguous()
        x = x.view(batch_size * chunk_size, inter_length, num_features)
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.view(batch_size, chunk_size * inter_length, num_features)
        x = x.permute(0, 2, 1).contiguous()

        if self.norm is not None:
            x = self.norm(x)

        x = x.view(batch_size, num_features, chunk_size, inter_length)
        x = x.permute(0, 1, 3, 2).contiguous()

        output = x + residual

        return output


def get_rnn(
    rnn: Union[str, nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]],
    input_size: Optional[int] = None,
    hidden_size: Optional[int] = None,
    batch_first: bool = False,
    bidirectional: bool = False,
    **kwargs,
) -> Union[nn.Module, Callable[[torch.Tensor], Tuple[torch.Tensor, Any]]]:
    if isinstance(rnn, nn.Module):
        rnn = copy.deepcopy(rnn)

    if callable(rnn):
        return rnn

    assert isinstance(rnn, str)
    assert input_size is not None
    assert hidden_size is not None

    rnn_kwargs = {
        "batch_first": batch_first,
        "bidirectional": bidirectional,
    }
    rnn_kwargs.update(kwargs)

    if rnn.lower() == "rnn":
        rnn = nn.RNN(input_size, hidden_size, **rnn_kwargs)
    elif rnn.lower() == "lstm":
        rnn = nn.LSTM(input_size, hidden_size, **rnn_kwargs)
    elif rnn.lower() == "gru":
        rnn = nn.GRU(input_size, hidden_size, **rnn_kwargs)
    else:
        raise NotImplementedError(f"{rnn} is not supported as rnn.")

    return rnn
