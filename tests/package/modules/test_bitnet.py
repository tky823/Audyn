import pytest
import torch

from audyn.modules.bitnet import (
    BitLinear158,
    BitLinear158Inference,
    BitMultiheadAttention158,
    BitMultiheadAttention158Inference,
)


@pytest.mark.parametrize("bias", [True, False])
def test_bitlinear158(bias: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    in_features, out_features = 4, 2
    length = 9

    # w/o group_dim
    module = BitLinear158(in_features, out_features, bias=bias)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)
    loss = torch.mean(output**2)
    loss.backward()

    assert output.size() == (batch_size, length, out_features)

    for p in module.parameters():
        if p.requires_grad:
            assert p.grad is not None

    # w/ group_dim: batch-wise scaling
    module = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        group_dim=(1, -1),
    )

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    assert output.size() == (batch_size, length, out_features)

    recomputed_input, _ = torch.split(input, [batch_size - 2, 2], dim=0)
    recomputed_output = module(recomputed_input)
    output, _ = torch.split(output, [batch_size - 2, 2], dim=0)

    assert torch.allclose(recomputed_output, output)

    # w/ group_dim: batch and token wise scaling
    module = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        group_dim=-1,
    )

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    assert output.size() == (batch_size, length, out_features)

    recomputed_input, _ = torch.split(input, [batch_size - 2, 2], dim=0)
    recomputed_input, _ = torch.split(recomputed_input, [length // 2, length - length // 2], dim=1)
    recomputed_output = module(recomputed_input)
    output, _ = torch.split(output, [batch_size - 2, 2], dim=0)
    output, _ = torch.split(output, [length // 2, length - length // 2], dim=1)

    assert torch.allclose(recomputed_output, output)


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

    # w/ group_dim: batch-wise scaling
    if batch_first:
        group_dim = (1, -1)
    else:
        group_dim = (0, -1)

    module = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
        group_dim=group_dim,
    )
    module.eval()

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

    if batch_first:
        batch_dim = 0
    else:
        batch_dim = 1

    recomputed_query, _ = torch.split(query, [batch_size - 2, 2], dim=batch_dim)
    recomputed_key, _ = torch.split(key, [batch_size - 2, 2], dim=batch_dim)
    recomputed_value, _ = torch.split(value, [batch_size - 2, 2], dim=batch_dim)

    recomputed_output, recomputed_attn_weights = module(
        recomputed_query, recomputed_key, recomputed_value
    )
    output, _ = torch.split(output, [batch_size - 2, 2], dim=batch_dim)
    attn_weights, _ = torch.split(attn_weights, [batch_size - 2, 2], dim=0)

    assert torch.allclose(recomputed_output, output)
    assert torch.allclose(recomputed_attn_weights, attn_weights)

    # w/ group_dim: batch and token wise scaling
    module = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
        group_dim=-1,
    )
    module.eval()

    query = torch.randn((query_length, batch_size, embed_dim))
    key = torch.randn((key_length, batch_size, embed_dim))
    value = torch.randn((key_length, batch_size, embed_dim))

    if batch_first:
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)

    key_padding_mask = torch.arange(key_length) >= key_length // 2
    key_padding_mask = key_padding_mask.expand((batch_size, key_length))
    output, attn_weights = module(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
    )

    if batch_first:
        assert output.size() == (batch_size, query_length, embed_dim)
    else:
        assert output.size() == (query_length, batch_size, embed_dim)

    assert attn_weights.size() == (batch_size, query_length, key_length)

    if batch_first:
        batch_dim = 0
        length_dim = 1
    else:
        batch_dim = 1
        length_dim = 0

    recomputed_query, _ = torch.split(query, [batch_size - 2, 2], dim=batch_dim)
    recomputed_key, _ = torch.split(key, [batch_size - 2, 2], dim=batch_dim)
    recomputed_value, _ = torch.split(value, [batch_size - 2, 2], dim=batch_dim)
    recomputed_query, _ = torch.split(
        recomputed_query, [query_length // 2, query_length - query_length // 2], dim=length_dim
    )
    recomputed_key, _ = torch.split(
        recomputed_key, [key_length // 2, key_length - key_length // 2], dim=length_dim
    )
    recomputed_value, _ = torch.split(
        recomputed_value, [key_length // 2, key_length - key_length // 2], dim=length_dim
    )
    key_padding_mask, _ = torch.split(key_padding_mask, [batch_size - 2, 2], dim=0)
    key_padding_mask, _ = torch.split(
        key_padding_mask, [key_length // 2, key_length - key_length // 2], dim=-1
    )

    recomputed_output, recomputed_attn_weights = module(
        recomputed_query,
        recomputed_key,
        recomputed_value,
        key_padding_mask=key_padding_mask,
    )
    output, _ = torch.split(output, [batch_size - 2, 2], dim=batch_dim)
    attn_weights, _ = torch.split(attn_weights, [batch_size - 2, 2], dim=0)
    output, _ = torch.split(
        output, [query_length // 2, query_length - query_length // 2], dim=length_dim
    )
    attn_weights, _ = torch.split(
        attn_weights, [query_length // 2, query_length - query_length // 2], dim=1
    )
    attn_weights, _ = torch.split(
        attn_weights, [key_length // 2, key_length - key_length // 2], dim=2
    )

    assert torch.allclose(recomputed_output, output)
    assert torch.allclose(recomputed_attn_weights, attn_weights)


@pytest.mark.parametrize("bias", [True, False])
def test_bitlinear158_inference(bias: bool) -> None:
    torch.manual_seed(0)

    batch_size = 5
    in_features, out_features = 4, 2
    length = 9

    module = BitLinear158(in_features, out_features, bias=bias)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    inference_module = BitLinear158Inference.build_from_bitlinear158(module)
    inference_output = inference_module(input)

    assert torch.allclose(inference_output, output)

    # w/ group_dim: batch-wise scaling
    module = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        group_dim=(1, -1),
    )
    module = BitLinear158Inference.build_from_bitlinear158(module)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    assert output.size() == (batch_size, length, out_features)

    recomputed_input, _ = torch.split(input, [batch_size - 2, 2], dim=0)
    recomputed_output = module(recomputed_input)
    output, _ = torch.split(output, [batch_size - 2, 2], dim=0)

    assert torch.allclose(recomputed_output, output)

    # w/ group_dim: batch and token wise scaling
    module = BitLinear158(
        in_features,
        out_features,
        bias=bias,
        group_dim=-1,
    )
    module = BitLinear158Inference.build_from_bitlinear158(module)

    input = torch.randn((batch_size, length, in_features))
    output = module(input)

    assert output.size() == (batch_size, length, out_features)

    recomputed_input, _ = torch.split(input, [batch_size - 2, 2], dim=0)
    recomputed_input, _ = torch.split(recomputed_input, [length // 2, length - length // 2], dim=1)
    recomputed_output = module(recomputed_input)
    output, _ = torch.split(output, [batch_size - 2, 2], dim=0)
    output, _ = torch.split(output, [length // 2, length - length // 2], dim=1)

    assert torch.allclose(recomputed_output, output)


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("batch_first", [True, False])
def test_bitmha158_inference(bias: bool, batch_first: bool) -> None:
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
    module.eval()

    query = torch.randn((query_length, batch_size, embed_dim))
    key = torch.randn((key_length, batch_size, embed_dim))
    value = torch.randn((key_length, batch_size, embed_dim))

    if batch_first:
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)

    output, attn_weights = module(query, key, value)

    inference_module = BitMultiheadAttention158Inference.build_from_bitmha158(module)
    inference_module.eval()
    inference_output, inference_attn_weights = inference_module(query, key, value)

    assert torch.allclose(output, inference_output)
    assert torch.allclose(attn_weights, inference_attn_weights)

    # w/ group_dim: batch-wise scaling
    if batch_first:
        group_dim = (1, -1)
    else:
        group_dim = (0, -1)

    module = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
        group_dim=group_dim,
    )
    module = BitMultiheadAttention158Inference.build_from_bitmha158(module)
    module.eval()

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

    if batch_first:
        batch_dim = 0
    else:
        batch_dim = 1

    recomputed_query, _ = torch.split(query, [batch_size - 2, 2], dim=batch_dim)
    recomputed_key, _ = torch.split(key, [batch_size - 2, 2], dim=batch_dim)
    recomputed_value, _ = torch.split(value, [batch_size - 2, 2], dim=batch_dim)

    recomputed_output, recomputed_attn_weights = module(
        recomputed_query, recomputed_key, recomputed_value
    )
    output, _ = torch.split(output, [batch_size - 2, 2], dim=batch_dim)
    attn_weights, _ = torch.split(attn_weights, [batch_size - 2, 2], dim=0)

    assert torch.allclose(recomputed_output, output)
    assert torch.allclose(recomputed_attn_weights, attn_weights)

    # w/ group_dim: batch and token wise scaling
    module = BitMultiheadAttention158(
        embed_dim,
        num_heads,
        bias=bias,
        batch_first=batch_first,
        group_dim=-1,
    )
    module = BitMultiheadAttention158Inference.build_from_bitmha158(module)
    module.eval()

    query = torch.randn((query_length, batch_size, embed_dim))
    key = torch.randn((key_length, batch_size, embed_dim))
    value = torch.randn((key_length, batch_size, embed_dim))

    if batch_first:
        query = query.transpose(1, 0)
        key = key.transpose(1, 0)
        value = value.transpose(1, 0)

    key_padding_mask = torch.arange(key_length) >= key_length // 2
    key_padding_mask = key_padding_mask.expand((batch_size, key_length))
    output, attn_weights = module(
        query,
        key,
        value,
        key_padding_mask=key_padding_mask,
    )

    if batch_first:
        assert output.size() == (batch_size, query_length, embed_dim)
    else:
        assert output.size() == (query_length, batch_size, embed_dim)

    assert attn_weights.size() == (batch_size, query_length, key_length)

    if batch_first:
        batch_dim = 0
        length_dim = 1
    else:
        batch_dim = 1
        length_dim = 0

    recomputed_query, _ = torch.split(query, [batch_size - 2, 2], dim=batch_dim)
    recomputed_key, _ = torch.split(key, [batch_size - 2, 2], dim=batch_dim)
    recomputed_value, _ = torch.split(value, [batch_size - 2, 2], dim=batch_dim)
    recomputed_query, _ = torch.split(
        recomputed_query, [query_length // 2, query_length - query_length // 2], dim=length_dim
    )
    recomputed_key, _ = torch.split(
        recomputed_key, [key_length // 2, key_length - key_length // 2], dim=length_dim
    )
    recomputed_value, _ = torch.split(
        recomputed_value, [key_length // 2, key_length - key_length // 2], dim=length_dim
    )
    key_padding_mask, _ = torch.split(key_padding_mask, [batch_size - 2, 2], dim=0)
    key_padding_mask, _ = torch.split(
        key_padding_mask, [key_length // 2, key_length - key_length // 2], dim=-1
    )

    recomputed_output, recomputed_attn_weights = module(
        recomputed_query,
        recomputed_key,
        recomputed_value,
        key_padding_mask=key_padding_mask,
    )
    output, _ = torch.split(output, [batch_size - 2, 2], dim=batch_dim)
    attn_weights, _ = torch.split(attn_weights, [batch_size - 2, 2], dim=0)
    output, _ = torch.split(
        output, [query_length // 2, query_length - query_length // 2], dim=length_dim
    )
    attn_weights, _ = torch.split(
        attn_weights, [query_length // 2, query_length - query_length // 2], dim=1
    )
    attn_weights, _ = torch.split(
        attn_weights, [key_length // 2, key_length - key_length // 2], dim=2
    )

    assert torch.allclose(recomputed_output, output)
    assert torch.allclose(recomputed_attn_weights, attn_weights)
