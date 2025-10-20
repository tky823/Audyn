"""This file is based on
https://github.com/pytorch/pytorch/blob/0093df78df590a35deb784773aa2165884c1b7bd/torch/optim/optimizer.py.
"""

import copy
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Type,
    Union,
)

import torch
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle


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
            >>> from audyn.optim.optimizer import MovingAverageWrapper
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

        def _pack_param_group(param_group):
            nonlocal start_index
            param_mappings.update(
                {
                    id(p): i
                    for i, p in enumerate(param_group["params"], start_index)
                    if id(p) not in param_mappings
                }
            )
            packed = {"params": [param_mappings[id(p)] for p in param_group["params"]]}
            start_index += len(packed["params"])

            return packed

        param_groups = [
            _pack_param_group(param_group) for param_group in self.moving_average_param_groups
        ]
        packed_state = {}

        for group in self.moving_average_param_groups:
            for k, params in group.items():
                assert k == "params", "Only params is supported."

                for p in params:
                    assert isinstance(p, torch.Tensor), (
                        "Only torch.Tensor is supported, but found {}.".format(type(p))
                    )

                    packed_state.update({param_mappings[id(p)]: p.data})

        packed_state["smooth"] = self.smooth

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

        # for backward compatibility
        self.smooth = moving_averate_state_dict["state"].get("smooth", 0.999)

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

    # define methods as those of self.optimizer
    def __getstate__(self) -> Dict[str, Any]:
        return self.optimizer.__getstate__()

    def __setstate__(self, state: Dict[str, Any]) -> None:
        return self.optimizer.__setstate__(state)

    def __repr__(self) -> str:
        return self.optimizer.__repr__()

    def _cuda_graph_capture_health_check(self) -> None:
        return self.optimizer._cuda_graph_capture_health_check()

    def _optimizer_step_code(self) -> None:
        return self.optimizer._optimizer_step_code()

    def _patch_step_function(self) -> None:
        return self.optimizer._patch_step_function()

    def register_step_pre_hook(self, hook: Callable[[Any], None]) -> RemovableHandle:
        return self.optimizer.register_step_pre_hook(hook)

    def register_step_post_hook(self, hook: Callable[[Any], None]) -> RemovableHandle:
        return self.optimizer.register_step_post_hook(hook)

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        return self.optimizer.add_param_group(param_group)

    # if not found __name as attribute of self, search self.optimizer instead.
    def __getattr__(self, __name: str) -> Any:
        return getattr(self.optimizer, __name)


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

    def step(self, *args, **kwargs) -> None:
        """Performs a single optimization step and update exponential moving average."""
        self.optimizer.step(*args, **kwargs)

        for param_group, moving_average_param_group in zip(
            self.optimizer.param_groups, self.moving_average_param_groups
        ):
            for p, p_moving_average in zip(
                param_group["params"], moving_average_param_group["params"]
            ):
                p_moving_average.data = torch.lerp(
                    p.data, p_moving_average.data, weight=self.smooth
                )


class MultiOptimizers:
    """Module to manage multiple optimizers.

    .. note::

        To use this class with learning scheduler, you have to choose MultiLRSchedulers.

    """

    # TODO: improve design

    def __init__(self, optimizers: List[Union[Dict[str, Any], Optimizer]]) -> None:
        self.optimizers = {}

        for idx, optimizer in enumerate(optimizers):
            if isinstance(optimizer, Optimizer):
                k = str(idx)
                v = optimizer
            elif isinstance(optimizer, dict):
                k = optimizer["name"]
                v = optimizer["optimizer"]
            else:
                raise ValueError(f"{type(optimizer)} is not supported.")

            if k in self.optimizers.keys():
                raise ValueError(f"Duplicate optimizer name {k} is found.")

            self.optimizers[k] = v

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers.values():
            optimizer: Optimizer
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers.values():
            optimizer: Optimizer
            optimizer.step(*args, **kwargs)

    def state_dict(self, *args, **kwargs) -> Dict[str, Dict[str, Any]]:
        state_dict = {}

        for name, optimizer in self.optimizers.items():
            optimizer: Optimizer
            state_dict[name] = optimizer.state_dict(*args, **kwargs)

        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Dict[str, Any]]) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to ``state_dict``.

        """

        for name, optimizer in self.optimizers.items():
            optimizer: Optimizer
            optimizer.load_state_dict(state_dict[name])


class GANOptimizer:
    def __init__(self, generator: Optimizer, discriminator: Optimizer) -> None:
        self.generator = generator
        self.discriminator = discriminator

    def zero_grad(self, *args, **kwargs) -> None:
        self.generator.zero_grad(*args, **kwargs)
        self.discriminator.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs) -> None:
        self.generator.step(*args, **kwargs)
        self.discriminator.step(*args, **kwargs)
