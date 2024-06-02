import pytest
import torch

from audyn.modules.bitnet import BitLinearB158


@pytest.mark.parametrize("bias", [True, False])
def test_bitlinear158(bias: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    in_features, out_features = 4, 2
    length = 9

    module = BitLinearB158(in_features, out_features, bias=bias)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    assert output.size() == (batch_size, length, out_features)
