import pytest
import torch
from audyn_test import allclose

from audyn.functional.hyperbolic import mobius_add, mobius_scalar_mul, mobius_sub


@pytest.mark.parametrize("curvature", [-1, -2])
def test_mobius_add(curvature: float) -> None:
    torch.manual_seed(0)

    batch_size = 10
    num_features = 5

    input = torch.rand((batch_size, num_features)) - 0.5
    other = torch.rand((batch_size, num_features)) - 0.5

    output_by_scaler_curvature = mobius_add(input, other, curvature=curvature)

    curvature0 = torch.tensor([curvature])
    curvature1 = 2 * torch.rand((batch_size - 1,)) - 3
    curvature_by_tensor = torch.cat([curvature0, curvature1], dim=0)

    output_by_tensor_curvature = mobius_add(input, other, curvature=curvature_by_tensor)

    output_by_scaler_curvature, *_ = torch.unbind(output_by_scaler_curvature, dim=0)
    output_by_tensor_curvature, *_ = torch.unbind(output_by_tensor_curvature, dim=0)

    allclose(output_by_scaler_curvature, output_by_tensor_curvature)

    input = torch.rand((batch_size, num_features)) - 0.5
    other = torch.rand(()).item() - 0.5
    output_by_scaler = mobius_add(input, other, curvature=curvature)

    other = other * torch.ones((num_features,), dtype=input.dtype)
    output_by_tensor = mobius_add(input, other, curvature=curvature)

    allclose(output_by_scaler, output_by_tensor)

    other = torch.rand((batch_size, num_features)) - 0.5
    input = torch.rand(()).item() - 0.5
    output_by_scaler = mobius_add(input, other, curvature=curvature)

    input = input * torch.ones((num_features,), dtype=other.dtype)
    output_by_tensor = mobius_add(input, other, curvature=curvature)

    allclose(output_by_scaler, output_by_tensor)

    input = torch.rand(()).item() - 0.5
    other = torch.rand(()).item() - 0.5
    output_by_scaler = mobius_add(input, other, curvature=curvature)

    input = torch.tensor([input], dtype=torch.float)
    other = torch.tensor([other], dtype=torch.float)
    output_by_tensor = mobius_add(input, other, curvature=curvature)

    allclose(output_by_scaler, output_by_tensor)


@pytest.mark.parametrize("curvature", [-1, -2])
def test_mobius_sub(curvature: float) -> None:
    torch.manual_seed(0)

    batch_size = 10
    num_features = 5

    input = torch.rand((batch_size, num_features)) - 0.5
    other = torch.rand((batch_size, num_features)) - 0.5

    output_by_scaler_curvature = mobius_sub(input, other, curvature=curvature)

    curvature0 = torch.tensor([curvature])
    curvature1 = 2 * torch.rand((batch_size - 1,)) - 3
    curvature = torch.cat([curvature0, curvature1], dim=0)

    output_by_tensor_curvature = mobius_sub(input, other, curvature=curvature)

    output_by_scaler_curvature, *_ = torch.unbind(output_by_scaler_curvature, dim=0)
    output_by_tensor_curvature, *_ = torch.unbind(output_by_tensor_curvature, dim=0)

    allclose(output_by_scaler_curvature, output_by_tensor_curvature)


@pytest.mark.parametrize("curvature", [-1, -2])
def test_mobius_scalar_mul(curvature: float) -> None:
    torch.manual_seed(0)

    batch_size = 10
    num_features = 5

    input = torch.rand((batch_size, num_features)) - 0.5

    output_by_add = mobius_add(input, input, curvature=curvature)
    output_by_add = mobius_add(output_by_add, input, curvature=curvature)
    output = mobius_scalar_mul(input, 3, curvature=curvature)

    allclose(output, output_by_add, atol=1e-4)
