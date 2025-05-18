import torch


def allclose(
    input: torch.Tensor,
    other: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> None:
    """Wrapper of torch.allclose.

    Raises:
        AssertionError: When torch.allclose yields False.

    """
    loss_max = torch.abs(input - other).max()

    assert torch.allclose(
        input, other, rtol=rtol, atol=atol, equal_nan=equal_nan
    ), f"Max absolute error: {loss_max}."
