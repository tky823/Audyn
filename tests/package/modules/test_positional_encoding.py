import pytest
import torch

from audyn.modules.positional_encoding import AbsolutePositionalEncoding, RotaryPositionalEmbedding

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


@pytest.mark.parametrize("batch_first", parameters_batch_first)
def test_rotary_positional_embedding(batch_first: bool):
    torch.manual_seed(0)

    batch_size = 2
    length = 8
    embed_dim = 10

    positional_encoding = RotaryPositionalEmbedding(embed_dim, batch_first=batch_first)

    input = torch.randn((batch_size, length, embed_dim))

    if not batch_first:
        input = input.permute(1, 0, 2)

    output = positional_encoding(input)

    assert output.size() == input.size()

    # ensure property
    q = torch.randn((batch_size, 1, embed_dim))
    k = torch.randn((batch_size, 1, embed_dim))

    q = q.expand(batch_size, length, embed_dim)
    k = k.expand(batch_size, length, embed_dim)

    if not batch_first:
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)

    q = positional_encoding(q)
    k = positional_encoding(k)

    if batch_first:
        k = k.transpose(-2, -1)
    else:
        q = q.permute(1, 0, 2)
        k = k.permute(1, 2, 0)

    qk = torch.matmul(q, k)

    for offset in range(length):
        qk_diag = torch.diagonal(qk, offset=offset, dim1=-2, dim2=-1)
        qk_mean = qk_diag.mean(dim=1, keepdim=True)

        assert torch.allclose(qk_diag, qk_mean, atol=1e-5)

        qk_diag = torch.diagonal(qk, offset=-offset, dim1=-2, dim2=-1)
        qk_mean = qk_diag.mean(dim=1, keepdim=True)

        assert torch.allclose(qk_diag, qk_mean, atol=1e-5)
