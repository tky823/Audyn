from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["AdditiveCoupling", "AffineCoupling", "ChannelSplitFlow"]


class BaseFlow(nn.Module):
    """Base class of flow."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, *args, logdet: torch.Tensor = None, reverse: bool = False
    ) -> Union[Any, Tuple[Any, torch.Tensor]]:
        raise NotImplementedError(f"Implement forward pass of {self.__class__.__name__}.")

    def initialize_logdet_if_necessary(
        self, logdet: torch.Tensor = None, device: torch.device = None
    ) -> torch.Tensor:
        if type(logdet) is int or type(logdet) is float:
            logdet = torch.full((), fill_value=logdet, device=device)

        return logdet


class AdditiveCoupling(BaseFlow):
    def __init__(self, split_channels: List[int], coupling: nn.Module) -> None:
        super().__init__()

        self.split_channels = split_channels
        self.coupling = coupling

    def forward(
        self, input: torch.Tensor, logdet: Optional[torch.Tensor] = None, reverse: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Forward pass of additive coupling.

        Args:
            input (torch.Tensor): Tensor of shape (\*, sum(split_channels)).
            logdet (torch.Tensor, optional): Log-determinant.
            reverse (bool): If ``reverse=True``, reverse process is used.

        Returns:
            torch.Tensor: Tensor of shape (\*, sum(split_channels)).

        """
        assert input.dim() == 2

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if reverse:
            y1, y2 = torch.split(input, self.split_channels, dim=-1)
            m_y1 = self.coupling(y1)

            x1 = y1
            x2 = y2 - m_y1

            output = torch.cat([x1, x2], dim=-1)
        else:
            x1, x2 = torch.split(input, self.split_channels, dim=-1)
            m_x1 = self.coupling(x1)

            y1 = x1
            y2 = x2 + m_x1

            output = torch.cat([y1, y2], dim=-1)

        if logdet is None:
            return output
        else:
            return output, logdet


class AffineCoupling(BaseFlow):
    def __init__(
        self,
        coupling: nn.Module,
        split: Optional[nn.Module] = None,
        concat: Optional[nn.Module] = None,
        scaling: bool = False,
        scaling_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.split = split
        self.coupling = coupling
        self.concat = concat

        if scaling:
            if scaling_channels is None:
                # Scalar scaling factor
                self.scaling_factor = nn.Parameter(
                    torch.empty((), dtype=torch.float), requires_grad=True
                )
            else:
                self.scaling_factor = nn.Parameter(
                    torch.empty(scaling_channels, dtype=torch.float), requires_grad=True
                )
        else:
            self.register_buffer("scaling_factor", None)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.scaling_factor is not None:
            self.scaling_factor.data.zero_()

    def forward(
        self,
        input: torch.Tensor,
        logdet: torch.Tensor = None,
        reverse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert (
            input.dim() == 2
        ), f"'input' is expected to be 2-D tensor, but given {input.dim()}-D tensor."

        logdet = self.initialize_logdet_if_necessary(logdet, device=input.device)

        if reverse:
            y1, y2 = self.call_split(input)
            log_s, t = self.coupling(y1)

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            x1 = y1
            x2 = (y2 - t) * torch.exp(-log_s)
            output = self.call_concat(x1, x2)

            if logdet is not None:
                logdet = logdet - log_s.sum(dim=-1)
        else:
            x1, x2 = self.call_split(input)
            log_s, t = self.coupling(x1)

            if self.scaling_factor is not None:
                scale = torch.exp(self.scaling_factor)
                log_s = torch.tanh(log_s / scale) * scale

            y1 = x1
            y2 = x2 * torch.exp(log_s) + t
            output = self.call_concat(y1, y2)

            if logdet is not None:
                logdet = logdet + log_s.sum(dim=-1)

        if logdet is None:
            return output
        else:
            return output, logdet

    def call_split(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.split is None:
            num_features = input.size(-1)
            output = torch.split(
                input, [num_features // 2, num_features - num_features // 2], dim=-1
            )
        else:
            output = self.split(input)

        return output

    def call_concat(self, input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.concat is None:
            output = torch.cat([input, other], dim=-1)
        else:
            output = self.concat(input, other)

        return output


class ChannelSplitFlow(BaseFlow):
    def __init__(self, channels: List[int]) -> None:
        super().__init__()

        self.channels = channels

    def forward(self, *args, logdet: torch.Tensor = None, reverse: bool = False) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        if reverse:
            assert len(args) == len(self.channels), "Length of arguments is invalid."

            output = torch.cat(args, dim=1)
        else:
            assert len(args) == 1, "Too long argument is given."

            output = torch.split(args[0], self.channels, dim=1)

        if logdet is None:
            return output
        else:
            return output, logdet
