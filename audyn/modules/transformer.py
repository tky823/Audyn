import torch.nn as nn

__all__ = [
    "get_activation",
]


def get_activation(activation: str) -> nn.Module:
    """Get activation module by str.

    Args:
        activation (str): Name of activation module.

    Returns:
        nn.Module: Activation module.

    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "elu":
        return nn.ELU()

    raise RuntimeError(f"activation should be relu/gelu/elu, not {activation}")
