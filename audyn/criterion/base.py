from typing import Any, Dict

import torch.nn as nn


class BaseCriterionWrapper(nn.Module):
    """Wrapper class to handle multiple criteria by MultiCriteria."""

    def __init__(
        self,
        criterion: nn.Module,
        key_mapping: Dict[str, Any],
        weight: float = 1,
    ) -> None:
        super().__init__()

        self.criterion = criterion
        self.key_mapping = key_mapping
        self.weight = weight

    def forward(self, *args, **kwargs) -> Any:
        return self.criterion(*args, **kwargs)


class MultiCriteria(nn.ModuleDict):
    """Base class of dict-type multiple criteria.

    Args:
        kwargs: Keyword arguments. Each item should inherit ``BaseCriterion``.

    Examples:

        >>> import audyn
        >>> import torch
        >>> torch.manual_seed(0)
        >>> config = {
        ...     "_target_": "audyn.criterion.MultiCriteria",
        ...     "mse": {
        ...         "_target_": "audyn.criterion.BaseCriterionWrapper",
        ...         "criterion": {
        ...             "_target_": "torch.nn.MSELoss",
        ...             "reduction": "mean",
        ...         },
        ...         "weight": 1,
        ...         "key_mapping": {
        ...             "estimated": {
        ...                 "input": "y",
        ...             },
        ...             "target": {
        ...                 "target": "t_mse",
        ...             }
        ...         }
        ...     },
        ...     "mae": {
        ...         "_target_": "audyn.criterion.BaseCriterionWrapper",
        ...         "criterion": {
        ...             "_target_": "torch.nn.L1Loss",
        ...             "reduction": "mean",
        ...         },
        ...         "weight": 2,
        ...         "key_mapping": {
        ...             "estimated": {
        ...                 "input": "y",
        ...             },
        ...             "target": {
        ...                 "target": "t_mae",
        ...             }
        ...         }
        ...     }
        >>> }
        >>> criterion = audyn.utils.instantiate_criterion(config)
        >>> y = torch.randn((4,))
        >>> t_mse = torch.randn_like(y)
        >>> t_mae = torch.randn_like(y)
        >>> criterion["mse"].weight * criterion["mse"](input=y, target=t_mse) \
        ...     + criterion["mae"].weight * criterion["mae"](input=y, target=t_mae)
        tensor(5.8831)

    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        for k, v in kwargs.items():
            assert isinstance(k, str), f"Invalid key {k} is found."
            assert callable(v), "Criterion should be callable."

        super().__init__(kwargs)

    def forward(self, *args, **kwargs) -> None:
        raise NotImplementedError("forward method is not supported.")
