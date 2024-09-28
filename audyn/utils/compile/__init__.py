# module for torch.compile

import torch.nn as nn

__all__ = [
    "is_compiled_module",
]


def is_compiled_module(module: nn.Module) -> bool:
    if hasattr(module, "_orig_mod"):
        # NOTE: isinstance(module, torch._dynamo.eval_frame.OptimizedModule)
        #       may be better.
        return True
    else:
        return False
