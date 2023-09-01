import pytest
import torch

from audyn.modules.positional_encoding import AbsolutePositionalEncoding

parameters_batch_first = [True, False]


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_absolute_positional_encoding(batch_first: bool):
    batch_size = 2
    length = 8
    embed_dim = 4

    positional_encoding = AbsolutePositionalEncoding(batch_first=batch_first)

    input = torch.randn((batch_size, length, embed_dim))

    if not batch_first:
        input = input.permute(1, 0, 2)

    output = positional_encoding(input)

    assert output.size() == input.size()
