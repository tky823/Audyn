import torch
import torch.nn as nn
import torch.nn.functional as F

from audyn.modules.normalization import MaskedLayerNorm


def test_masked_layer_norm() -> None:
    torch.manual_seed(0)

    batch_size, max_length, num_features = 4, 8, 3

    layer_norm = nn.LayerNorm((num_features,))
    masked_layer_norm = MaskedLayerNorm((num_features,))

    nn.init.normal_(layer_norm.weight.data)
    nn.init.normal_(layer_norm.bias.data)

    masked_layer_norm.weight.data.copy_(layer_norm.weight.data)
    masked_layer_norm.bias.data.copy_(layer_norm.bias.data)

    length = torch.randint(1, max_length, (batch_size,))
    max_length = torch.max(length).item()

    input = torch.randn((batch_size, max_length, num_features))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    masked_input = input.masked_fill(padding_mask.unsqueeze(dim=-1), 0)

    output = []

    for _input, _length in zip(input, length):
        _length = _length.item()
        _input = F.pad(_input, (0, 0, 0, _length - max_length))
        _output = layer_norm(_input)
        _output = F.pad(_output, (0, 0, 0, max_length - _length))
        output.append(_output)

    output = torch.stack(output, dim=0)

    masked_output = masked_layer_norm(masked_input, padding_mask=padding_mask.unsqueeze(dim=-1))

    assert torch.allclose(output, masked_output, atol=1e-6)

    input = torch.randn((batch_size, max_length, num_features))
    padding_mask = torch.zeros((batch_size, max_length, num_features), dtype=torch.bool)

    output = layer_norm(input)
    masked_output = masked_layer_norm(input, padding_mask=padding_mask)

    assert torch.allclose(output, masked_output, atol=1e-6)
