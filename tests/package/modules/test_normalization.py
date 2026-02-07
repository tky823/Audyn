import pytest
import torch
import torch.nn as nn


@pytest.mark.parametrize("batch_first", [True, False])
def test_masked_layer_norm(batch_first: bool) -> None:
    torch.manual_seed(0)

    batch_size, max_length, num_features = 4, 8, 3

    layer_norm = nn.LayerNorm((num_features,))

    nn.init.normal_(layer_norm.weight.data)
    nn.init.normal_(layer_norm.bias.data)

    length = torch.randint(1, max_length, (batch_size,))
    max_length = torch.max(length).item()

    input = torch.randn((batch_size, max_length, num_features))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    if not batch_first:
        input = input.permute(1, 0, 2)
        padding_mask = padding_mask.permute(1, 0)

    output = layer_norm(input)
    output = output.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

    masked_input = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)
    masked_output = layer_norm(masked_input)
    masked_output = masked_output.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

    assert torch.allclose(output, masked_output, atol=1e-6)
