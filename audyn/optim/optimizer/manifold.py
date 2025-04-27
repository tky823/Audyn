from typing import Any, Callable, Iterable, Optional, overload

import torch
import torch.nn as nn
from torch.optim import Optimizer

try:
    from torch.optim.optimizer import ParamsT

    optimizer_args_type = "ParamsT"

except ImportError:
    try:
        from torch.optim.optimizer import params_t

        optimizer_args_type = "params_t"
    except ImportError:
        optimizer_args_type = "Iterable"


class RiemannSGD(Optimizer):
    """Riemannian stochastic gradient descent.

    Examples:

        >>> import torch
        >>> from audyn.modules import PoincareEmbedding
        >>> from audyn.functional.poincare import poincare_distance
        >>> from audyn.criterion.negative_sampling import DistanceBasedNegativeSamplingLoss
        >>> from audyn.optim import RiemannSGD
        >>> num_embedings = 10
        >>> embedding_dim = 2
        >>> num_neg_samples = 5
        >>> manifold = PoincareEmbedding(num_embedings, embedding_dim)
        >>> criterion = DistanceBasedNegativeSamplingLoss(
        ...     poincare_distance,
        ...     positive_distance_kwargs={
        ...         "curvature": manifold.curvature,
        ...         "dim": -1,
        ...     },
        ...     negative_distance_kwargs={
        ...         "curvature": manifold.curvature,
        ...         "dim": -1,
        ...     },
        ... )
        >>> optimizer = RiemannSGD(
        ...     manifold.parameters(),
        ...     expmap=manifold.expmap,
        ...     proj=manifold.proj,
        ... )
        >>> anchor = torch.randint(0, num_embedings, (), dtype=torch.long)
        >>> positive = torch.randint(0, num_embedings, (), dtype=torch.long)
        >>> negative = torch.randint(0, num_embedings, (num_neg_samples,), dtype=torch.long)
        >>> anchor = manifold(anchor)
        >>> positive = manifold(positive)
        >>> negative = manifold(negative)
        >>> loss = criterion(anchor, positive, negative)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()

    """

    if optimizer_args_type == "ParamsT":

        @overload
        def __init__(
            self,
            params: ParamsT,
            lr: float = 1e-3,
            expmap: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ) -> None: ...

    elif optimizer_args_type == "params_t":

        @overload
        def __init__(
            self,
            params: params_t,
            lr: float = 1e-3,
            expmap: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ) -> None: ...

    else:

        @overload
        def __init__(
            self,
            params: Iterable,
            lr: float = 1e-3,
            expmap: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
            proj: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        ) -> None: ...

    def __init__(
        self,
        params,
        lr=1e-3,
        expmap=None,
        proj=None,
    ) -> None:
        defaults = dict(
            lr=lr,
        )

        super().__init__(params, defaults)

        self.expmap = expmap
        self.proj = proj

    def step(self, closure: Callable = None) -> Any:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for param_group in self.param_groups:
            lr = param_group["lr"]
            params: list[nn.Parameter] = param_group["params"]

            for param in params:
                grad = param.grad.data

                if self.expmap is None:
                    # i.e., retraction map
                    updated = -lr * grad + param.data
                else:
                    updated = self.expmap(-lr * grad, root=param.data)

                if self.proj is not None:
                    updated = self.proj(updated)

                param.data.copy_(updated)

        return loss
