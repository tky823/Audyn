import pytest
import torch

from audyn.modules.bitnet import BitLinear158, BitMultiheadAttention158


@pytest.mark.parametrize("bias", [True, False])
def test_bitlinear158(bias: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    in_features, out_features = 4, 2
    length = 9

    module = BitLinear158(in_features, out_features, bias=bias)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)
    loss = torch.mean(output**2)
    loss.backward()

    assert output.size() == (batch_size, length, out_features)

    for p in module.parameters():
        if p.requires_grad:
            assert p.grad is not None


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_bitmha158(bias: bool, batch_first: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    embed_dim, num_heads = 8, 4
    query_length, key_length = 7, 9

    module = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
    )

    query = torch.randn((query_length, batch_size, embed_dim))
    key = torch.randn((key_length, batch_size, embed_dim))
    value = torch.randn((key_length, batch_size, embed_dim))

    if batch_first:
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)

    output, attn_weights = module(query, key, value)
    loss = torch.mean(output**2)
    loss.backward()

    if batch_first:
        assert output.size() == (batch_size, query_length, embed_dim)
    else:
        assert output.size() == (query_length, batch_size, embed_dim)

    assert attn_weights.size() == (batch_size, query_length, key_length)

    for p in module.parameters():
        if p.requires_grad:
            assert p.grad is not None
