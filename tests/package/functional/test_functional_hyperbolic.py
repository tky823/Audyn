import pytest
import torch
from dummy import allclose

from audyn.functional.hyperbolic import mobius_add


@pytest.mark.parametrize("curvature", [-1, -2])
def test_mobius_add(curvature: float) -> None:
    torch.manual_seed(0)

    batch_size = 10
    num_features = 5

    input = torch.rand((batch_size, num_features)) - 0.5
    other = torch.rand((batch_size, num_features)) - 0.5

    output_by_scaler_curvature = mobius_add(input, other, curvature=curvature)

    curvature0 = torch.tensor([curvature])
    curvature1 = 2 * torch.rand((batch_size - 1,)) - 2
    curvature = torch.cat([curvature0, curvature1], dim=0)

    output_by_tensor_curvature = mobius_add(input, other, curvature=curvature)

    output_by_scaler_curvature, *_ = torch.unbind(output_by_scaler_curvature, dim=0)
    output_by_tensor_curvature, *_ = torch.unbind(output_by_tensor_curvature, dim=0)

    allclose(output_by_scaler_curvature, output_by_tensor_curvature)
