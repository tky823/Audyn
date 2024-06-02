import pytest
import torch

from audyn.modules.bitnet import BitLinearB158, BitMultiheadAttentionB158


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


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_bitmha158(bias: bool, batch_first: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    embed_dim, num_heads = 8, 4
    query_length, key_length = 7, 9

    module = BitMultiheadAttentionB158(
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

    if batch_first:
        assert output.size() == (batch_size, query_length, embed_dim)
    else:
        assert output.size() == (query_length, batch_size, embed_dim)

    assert attn_weights.size() == (batch_size, query_length, key_length)
