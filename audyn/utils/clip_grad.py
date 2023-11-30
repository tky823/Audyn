from typing import Iterable, overload

import torch
import torch.nn as nn
from packaging import version

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")

__all__ = ["GradClipper"]


class GradClipper:
    """Class to clip gradients.

    Args:
        params: Parameters to be clipped.
        mode (str): Clipping mode. ``value`` and ``norm`` are available.
        kwargs: Keyword arguments given to ``nn.utils.clip_grad_value_``
            or ``nn.utils.clip_grad_norm_``.

    Examples:

        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> batch_size = 4
        >>> in_features, out_features = 3, 2
        >>> clip_value = 0.1
        >>> model = nn.Linear(2, 3)
        >>> optimizer = Adam(model.parameters())
        >>> grad_clipper = GradClipper(
        ...     model.parameters(), mode="value", clip_value=clip_value
        ... )
        >>> input = torch.randn((batch_size, in_features))
        >>> target = torch.randn((batch_size, out_features))
        >>> output = model(input)
        >>> loss = torch.mean((output - target) ** 2)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> grad_clipper.step()
        >>> optimizer.step()

    """

    if IS_TORCH_LT_2_1:

        @overload
        def __init__(
            self,
            params: Iterable,
            mode: str = None,
            clip_value: float = None,
        ) -> None:
            ...

        @overload
        def __init__(
            self,
            params: Iterable,
            mode: str = None,
            max_norm: float = None,
            norm_type: float = 2,
            error_if_nonfinite: float = False,
        ) -> None:
            ...

    else:
        from torch.optim.optimizer import params_t

        @overload
        def __init__(
            self,
            params: params_t,
            mode: str = None,
            clip_value: float = None,
        ) -> None:
            ...

        @overload
        def __init__(
            self,
            params: params_t,
            mode: str = None,
            max_norm: float = None,
            norm_type: float = 2,
            error_if_nonfinite: float = False,
        ) -> None:
            ...

    def __init__(self, params, mode=None, **kwargs) -> None:
        if mode == "value":
            require_keys = {"clip_value"}
        elif mode == "norm":
            require_keys = {"max_norm"}
        else:
            raise ValueError(f"Clipping mode {mode} is not supported.")

        missing_keys = require_keys - set(kwargs.keys())

        if len(missing_keys) > 0:
            raise ValueError("Following keys are missing: {}.".format(missing_keys))

        if mode == "value":
            self.clip_value = kwargs["clip_value"]
            self.max_norm = None
            self.norm_type = None
            self.error_if_nonfinite = None
        elif mode == "norm":
            self.max_norm = kwargs["max_norm"]
            self.norm_type = kwargs.get("norm_type", 2)
            self.error_if_nonfinite = kwargs.get("error_if_nonfinite", False)
            self.clip_value = None
        else:
            raise ValueError(f"Clipping mode {mode} is not supported.")

        self.params = list(params)
        self.mode = mode

    def step(self) -> None:
        if self.mode == "value":
            nn.utils.clip_grad_value_(self.params, clip_value=self.clip_value)
        elif self.mode == "norm":
            nn.utils.clip_grad_norm_(
                self.params,
                max_norm=self.max_norm,
                norm_type=self.norm_type,
                error_if_nonfinite=self.error_if_nonfinite,
            )
        else:
            raise ValueError(f"Clipping mode {self.mode} is not supported.")
