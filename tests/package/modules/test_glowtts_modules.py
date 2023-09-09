import torch

from audyn.modules.glow import ActNorm1d
from audyn.modules.glowtts import MaskedActNorm1d


def test_masked_act_norm1d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features = 6
    max_length = 16

    # w/ 2D padding mask
    model = MaskedActNorm1d(num_features)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    non_padding_mask = torch.logical_not(padding_mask)
    num_elements_per_channel = non_padding_mask.sum()

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert z.size() == input.size()
    assert torch.allclose(output, input)
    assert torch.allclose(mean, torch.zeros(()), atol=1e-7)
    assert torch.allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    assert torch.allclose(output, input)
    assert torch.allclose(logdet, zeros)

    # w/ 3D padding mask
    batch_size = 4
    num_features = 3
    max_length = 6

    model = MaskedActNorm1d(num_features)

    length = torch.randint(
        num_features,
        num_features * max_length + 1,
        (batch_size,),
        dtype=torch.long,
    )
    max_length = torch.max(length)
    max_length = max_length + (num_features - max_length % num_features) % num_features
    input = torch.randn((batch_size, max_length))
    input = input.view(batch_size, num_features, -1)
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)
    padding_mask = padding_mask.view(batch_size, num_features, -1)
    non_padding_mask = torch.logical_not(padding_mask)
    num_elements_per_channel = non_padding_mask.sum(dim=(0, 2))

    input = input.masked_fill(padding_mask, 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert z.size() == input.size()
    assert torch.allclose(output, input)
    assert torch.allclose(mean, torch.zeros(()), atol=1e-7)
    assert torch.allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    output, logdet = model(
        z,
        padding_mask=padding_mask,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    assert torch.allclose(output, input)
    assert torch.allclose(logdet, zeros)

    # w/o padding mask
    batch_size = 2
    num_features = 6
    max_length = 16

    masked_model = MaskedActNorm1d(num_features)
    non_masked_model = ActNorm1d(num_features)

    input = torch.randn(batch_size, num_features, max_length)

    masked_z = masked_model(input)
    masked_output = masked_model(masked_z, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)

    assert torch.allclose(masked_z, non_masked_z)
    assert torch.allclose(masked_output, non_masked_output)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        logdet=zeros,
    )
    masked_output, masked_logdet = masked_model(
        masked_z,
        logdet=masked_z_logdet,
        reverse=True,
    )
    non_masked_z, non_masked_z_logdet = non_masked_model(
        input,
        logdet=zeros,
    )
    non_masked_output, non_masked_logdet = non_masked_model(
        non_masked_z,
        logdet=non_masked_z_logdet,
        reverse=True,
    )

    assert torch.allclose(masked_z, non_masked_z)
    assert torch.allclose(masked_output, non_masked_output)
    assert torch.allclose(masked_logdet, non_masked_logdet)
