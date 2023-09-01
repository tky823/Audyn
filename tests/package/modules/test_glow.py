import torch

from audyn.modules.glow import InvertiblePointwiseConv1d, InvertiblePointwiseConv2d


def test_invertible_pointwise_conv1d():
    torch.manual_seed(0)

    batch_size = 2
    num_features = 6
    length = 16

    model = InvertiblePointwiseConv1d(num_features)
    input = torch.randn(batch_size, num_features, length)

    z = model(input)
    output = model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    assert torch.allclose(output, input, atol=1e-6)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    assert torch.allclose(output, input, atol=1e-6)
    assert torch.allclose(logdet, zeros, atol=1e-7)


def test_invertible_pointwise_conv2d():
    torch.manual_seed(0)

    batch_size = 2
    num_features = 4
    height, width = 6, 6

    model = InvertiblePointwiseConv2d(num_features)
    input = torch.randn(batch_size, num_features, height, width)

    z = model(input)
    output = model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    e = torch.abs(output - input)
    assert torch.allclose(output, input, atol=1e-7), torch.max(e)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    assert torch.allclose(output, input, atol=1e-7)
    assert torch.allclose(logdet, zeros, atol=1e-7)
