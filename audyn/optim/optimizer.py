"""This file is based on
https://github.com/pytorch/pytorch/blob/0093df78df590a35deb784773aa2165884c1b7bd/torch/optim/optimizer.py.
"""
import copy
from typing import Any, Dict, Type

import torch
from torch.optim import Optimizer

__all__ = ["ExponentialMovingAverageWrapper"]


class MovingAverageWrapper(Optimizer):
    """Wrapper class of optimizer to perform moving average of parameters.

    Args:
        optimizer (Optimizer): Optimizer to update model parameters.
        smooth (float): Smoothing factor. Default: ``0.999``.

    Examples:

            >>> import torch
            >>> import torch.nn as nn
            >>> from torch.optim import Adam
            >>> from audyn.optim.optimizer import MovingAverageWrapper
            >>> in_channels, out_channels = 3, 2
            >>> lr = 1e-3
            >>> smooth = 0.999
            >>> model = nn.Linear(in_channels, out_channels)
            >>> optimizer = Adam(model.parameters(), lr=lr)
            >>> optimizer = MovingAverageWrapper(optimizer, smooth=smooth)
            >>> # or you can instantiate by build_from_optim_class
            >>> optimizer = MovingAverageWrapper.build_from_optim_class(
            ...     model.parameters(), optimizer_class=Adam, lr=lr, smooth=smooth
            ... )

    """

    def __init__(self, optimizer: Optimizer, smooth: float = 0.999) -> None:
        self.optimizer = optimizer
        self.smooth = smooth

        self.moving_average_param_groups = [
            {"params": copy.deepcopy(param_group["params"])}
            for param_group in self.optimizer.param_groups
        ]
        self.cached_param_groups = None

    def __getattr__(self, __name: str) -> Any:
        return getattr(self.optimizer, __name)

    @classmethod
    def build_from_optim_class(
        cls, *args, optimizer_class: Type, smooth: float = 0.999, **kwargs
    ) -> "MovingAverageWrapper":
        """Build moving average wrapper of specified optimizer.

        Args:
            optimizer_class (type): Optimizer class.
            smooth (float): Smoothing factor. Default: ``0.999``.
            args: Positional arguments given to instantiation of optimizer.
            kwargs: Keyword arguments given to instantiation of optimizer.

        Returns:
            MovingAverageWrapper: Moving average wrapper of optimizer.

        Examples:

            >>> import torch
            >>> import torch.nn as nn
            >>> from torch.optim import Adam
            >>> from veuth.optim.optimizer import MovingAverageWrapper
            >>> in_channels, out_channels = 3, 2
            >>> lr = 1e-3
            >>> smooth = 0.999
            >>> model = nn.Linear(in_channels, out_channels)
            >>> optimizer = MovingAverageWrapper.build_from_optim_class(
            ...     model.parameters(), optimizer_class=Adam, lr=lr, smooth=smooth
            ... )

        """
        optimizer = optimizer_class(*args, **kwargs)

        return cls(optimizer, smooth=smooth)

    def step(self) -> None:
        """Performs a single optimization step and update moving average."""
        raise NotImplementedError("`step` is not implemented.")

    def state_dict(self) -> Dict[str, Any]:
        """Returns the state of the optimizer as a ``dict``.

        Returns:
            dict: State dict of optimizer and moving average parameters.

        """
        state_dict = {}

        param_mappings = {}
        start_index = 0

        def pack_group(group):
            nonlocal start_index
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed = {"params": [param_mappings[id(p)] for p in group["params"]]}
            start_index += len(packed["params"])

            return packed

        param_groups = [pack_group(group) for group in self.moving_average_param_groups]
        # Remap state to use order indices as keys
        packed_state = {}

        for group in self.moving_average_param_groups:
            for k, v in group.items():
                assert k == "params", "Only params is supported."

                for p in v:
                    assert isinstance(
                        p, torch.Tensor
                    ), "Only torch.Tensor is supported, but found {}.".format(type(p))

                    packed_state.update({param_mappings[id(p)]: p.data})

        moving_averate_state_dict = {
            "state": packed_state,
            "param_groups": param_groups,
        }
        state_dict["moving_average"] = moving_averate_state_dict
        state_dict["original"] = self.optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to ``state_dict``.

        """
        moving_averate_state_dict = state_dict["moving_average"]
        optimizer_state_dict = state_dict["original"]

        # deepcopy, to be consistent with module API
        moving_averate_state_dict = copy.deepcopy(moving_averate_state_dict)
        # Validate the state_dict
        groups = self.moving_average_param_groups
        saved_packed_groups = moving_averate_state_dict["param_groups"]

        if len(groups) != len(saved_packed_groups):
            raise ValueError("Loaded state dict has a different number of parameter groups.")

        param_lens = (len(group["params"]) for group in groups)
        saved_lens = (len(group["params"]) for group in saved_packed_groups)

        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "Loaded state dict contains a parameter group "
                "that doesn't match the size of optimizer's group."
            )

        param_mappings = {}
        start_index = 0

        for group in groups:
            params = group["params"]
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(params, start_index)
                    if id(p) not in param_mappings
                }
            )
            packed_params = []

            for p in params:
                param_id = param_mappings[id(p)]
                packed_params.append(param_id)
                p.data = moving_averate_state_dict["state"][param_id]

            start_index += len(packed_params)

        # Load state dict of optimizer
        self.optimizer.load_state_dict(optimizer_state_dict)

    def set_moving_average_model(self) -> None:
        """Set moving averaged parameters to model."""
        if self.cached_param_groups is not None:
            raise ValueError("Call remove_moving_average_model before.")

        self.cached_param_groups = copy.deepcopy(self.optimizer.param_groups)

        for param_group, param_group_moving_average in zip(
            self.optimizer.param_groups, self.moving_average_param_groups
        ):
            for p, p_moving_average in zip(
                param_group["params"], param_group_moving_average["params"]
            ):
                p.data = p_moving_average.data

    def remove_moving_average_model(self) -> None:
        """Set original parameters to model."""
        if self.cached_param_groups is None:
            raise ValueError("Call set_moving_average_model before.")

        for param_group, cache_param_group in zip(
            self.optimizer.param_groups, self.cached_param_groups
        ):
            for p, p_cache in zip(param_group["params"], cache_param_group["params"]):
                p.data = p_cache.data

        self.cached_param_groups = None


class ExponentialMovingAverageWrapper(MovingAverageWrapper):
    """Wrapper class of optimizer to perform exponential moving average of parameters.

    Args:
        optimizer (Optimizer): Optimizer to update model parameters.
        smooth (float): Smoothing factor. Default: ``0.999``.

    Examples:

        >>> import torch
        >>> import torch.nn as nn
        >>> from torch.optim import Adam
        >>> from audyn.optim.optimizer import ExponentialMovingAverageWrapper
        >>> torch.manual_seed(0)
        >>> in_channels, out_channels = 3, 2
        >>> lr = 1e-3
        >>> smooth = 0.999
        >>> model = nn.Linear(in_channels, out_channels)
        >>> optimizer = Adam(model.parameters(), lr=lr)
        >>> optimizer = ExponentialMovingAverageWrapper(optimizer, smooth=smooth)
        >>> criterion = nn.MSELoss()
        >>> input, target = torch.randn(in_channels), torch.randn(out_channels)
        >>> output = model(input)
        >>> loss = criterion(output, target)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
        >>> model.weight
        Parameter containing:
        tensor([[-0.0053,  0.3087, -0.4742],
                [-0.4239, -0.2214,  0.1538]], requires_grad=True)
        >>> optimizer.set_moving_average_model()
        >>> model.weight
        Parameter containing:
        tensor([[-0.0043,  0.3097, -0.4752],
                [-0.4249, -0.2224,  0.1548]], requires_grad=True)
        >>> optimizer.remove_moving_average_model()
        >>> model.weight
        Parameter containing:
        tensor([[-0.0053,  0.3087, -0.4742],
                [-0.4239, -0.2214,  0.1538]], requires_grad=True)

    """

    def __init__(self, optimizer: Optimizer, smooth: float = 0.999) -> None:
        super().__init__(optimizer, smooth)

    def step(self) -> None:
        """Performs a single optimization step and update exponential moving average."""
        self.optimizer.step()

        for param_group, moving_average_param_group in zip(
            self.optimizer.param_groups, self.moving_average_param_groups
        ):
            for p, p_moving_average in zip(
                param_group["params"], moving_average_param_group["params"]
            ):
                p_moving_average.data = torch.lerp(
                    p.data, p_moving_average.data, weight=self.smooth
                )
