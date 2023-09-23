import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.modules.normalization import MaskedLayerNorm


def test_masked_layer_norm() -> None:
    torch.manual_seed(0)

    batch_size, length, num_features = 4, 8, 3

    input = torch.randn((batch_size, length, num_features))
    padding_mask = torch.zeros((batch_size, length, num_features), dtype=torch.bool)

    layer_norm = nn.LayerNorm((num_features,))
    masked_layer_norm = MaskedLayerNorm((num_features,))

    nn.init.normal_(layer_norm.weight.data)
    nn.init.normal_(layer_norm.bias.data)

    masked_layer_norm.weight.data.copy_(layer_norm.weight.data)
    masked_layer_norm.bias.data.copy_(layer_norm.bias.data)

    output = layer_norm(input)
    masked_output = masked_layer_norm(input, padding_mask=padding_mask)

    assert torch.allclose(output, masked_output)

    actual_length = length - 2
    masked_input = input
    input = F.pad(input, (0, 0, 0, actual_length - length))
    padding_mask = torch.arange(length) >= actual_length

    output = layer_norm(input)
    masked_output = masked_layer_norm(masked_input, padding_mask=padding_mask.unsqueeze(dim=-1))

    output = F.pad(output, (0, 0, 0, length - actual_length))

    assert torch.allclose(output, masked_output)
