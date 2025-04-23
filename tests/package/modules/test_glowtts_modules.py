import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from dummy import allclose
from omegaconf import OmegaConf

from audyn.modules.glow import ActNorm1d, InvertiblePointwiseConv1d
from audyn.modules.glowtts import (
    MaskedActNorm1d,
    MaskedInvertiblePointwiseConv1d,
    MaskedStackedResidualConvBlock1d,
    MaskedWaveNetAffineCoupling,
)
from audyn.modules.waveglow import WaveNetAffineCoupling


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
    allclose(output, input)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

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
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

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
    allclose(output, input)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

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
    mean = z.sum(dim=(0, 2)) / num_elements_per_channel
    std = torch.sum(z**2, dim=(0, 2)) / num_elements_per_channel

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

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
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

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
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    # w/ 2D padding mask, but all False (identical to w/o padding mask)
    masked_model = MaskedActNorm1d(num_features)
    non_masked_model = ActNorm1d(num_features)

    input = torch.randn(batch_size, num_features, max_length)
    padding_mask = torch.full((batch_size, max_length), fill_value=False)

    masked_z = masked_model(input, padding_mask=padding_mask)
    non_masked_z = non_masked_model(input)
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    non_masked_z, non_masked_z_logdet = non_masked_model(
        input,
        logdet=zeros,
    )
    mean = masked_z.sum(dim=(0, 2)) / (batch_size * max_length)
    std = torch.sum(masked_z**2, dim=(0, 2)) / (batch_size * max_length)

    allclose(masked_z, non_masked_z)
    allclose(masked_z_logdet, non_masked_z_logdet)
    allclose(mean, torch.zeros(()), atol=1e-7)
    allclose(std, torch.ones(()), atol=1e-7)


def test_masked_act_norm1d_ddp() -> None:
    """Ensure MaskedActNorm1d works well for DDP."""
    torch.manual_seed(0)

    port = str(torch.randint(0, 2**16, ()).item())
    world_size = 2
    seed = 0

    num_features = 6

    processes = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=train_dummy_masked_act_norm1d,
                args=(rank, world_size, port),
                kwargs={
                    "num_features": num_features,
                    "seed": seed,
                    "path": path,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        z = []

        rank = 0
        reference_model = MaskedActNorm1d(num_features)
        path = os.path.join(temp_dir, f"{rank}.pth")
        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
        reference_model.load_state_dict(state_dict["model"])
        latent = state_dict["latent"].permute(0, 2, 1)
        length = state_dict["length"]
        latent = nn.utils.rnn.unpad_sequence(latent, length, batch_first=True)
        latent = torch.cat(latent, dim=0)
        z.append(latent)

        for rank in range(1, world_size):
            model = MaskedActNorm1d(num_features)
            path = os.path.join(temp_dir, f"{rank}.pth")
            state_dict = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state_dict["model"])
            latent = state_dict["latent"].permute(0, 2, 1)
            length = state_dict["length"]
            latent = nn.utils.rnn.unpad_sequence(latent, length, batch_first=True)
            latent = torch.cat(latent, dim=0)
            z.append(latent)

            assert len(list(model.parameters())) == len(list(reference_model.parameters()))

            for param, param_reference in zip(model.parameters(), reference_model.parameters()):
                assert param.size() == param_reference.size()
                assert torch.equal(param, param_reference)

        std, mean = torch.std_mean(latent, dim=0, unbiased=False)

        allclose(mean, torch.zeros(()), atol=1e-7)
        allclose(std, torch.ones(()), atol=1e-7)


def test_masked_invertible_pointwise_conv1d() -> None:
    torch.manual_seed(0)

    batch_size = 2
    num_features, num_splits = 8, 4
    max_length = 16

    # w/ 2D padding mask
    model = MaskedInvertiblePointwiseConv1d(num_splits)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input, atol=1e-7)

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
    allclose(output, input, atol=1e-7)
    allclose(logdet, zeros)

    # w/o padding mask
    batch_size = 2
    num_features = num_splits = 6
    max_length = 16

    masked_model = MaskedInvertiblePointwiseConv1d(num_splits)
    non_masked_model = InvertiblePointwiseConv1d(num_features)
    non_masked_model.weight.data.copy_(masked_model.weight.data.detach())

    input = torch.randn(batch_size, num_features, max_length)

    masked_z = masked_model(input)
    masked_output = masked_model(masked_z, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)

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

    allclose(masked_z, non_masked_z)
    allclose(masked_z_logdet, non_masked_z_logdet)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)

    # w/ 2D padding mask, but all False (identical to w/o padding mask)
    masked_model = MaskedInvertiblePointwiseConv1d(num_splits)
    non_masked_model = InvertiblePointwiseConv1d(num_features)
    non_masked_model.weight.data.copy_(masked_model.weight.data.detach())

    input = torch.randn(batch_size, num_features, max_length)
    padding_mask = torch.full((batch_size, max_length), fill_value=False)

    masked_z = masked_model(input, padding_mask=padding_mask)
    masked_output = masked_model(masked_z, padding_mask=padding_mask, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    masked_output, masked_logdet = masked_model(
        masked_z,
        padding_mask=padding_mask,
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

    allclose(masked_z, non_masked_z)
    allclose(masked_z_logdet, non_masked_z_logdet)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)


def test_masked_wavenet_affine_coupling() -> None:
    torch.manual_seed(0)

    batch_size, max_length = 2, 16
    coupling_channels, hidden_channels = 4, 6
    num_layers = 3

    # w/ 2D padding mask
    model = MaskedWaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
    )

    nn.init.normal_(model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.coupling.bottleneck_conv1d_out.bias.data)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, 2 * coupling_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)

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
    allclose(output, input)
    allclose(logdet, zeros)

    # w/o padding mask
    masked_model = MaskedWaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
    )

    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.bias.data)

    input = torch.randn(batch_size, 2 * coupling_channels, max_length)
    z = masked_model(input)
    output = masked_model(z, reverse=True)

    assert output.size() == input.size()
    assert z.size() == input.size()
    allclose(output, input)

    zeros = torch.zeros((batch_size,))

    z, z_logdet = masked_model(
        input,
        logdet=zeros,
    )
    output, logdet = masked_model(
        z,
        logdet=z_logdet,
        reverse=True,
    )

    assert output.size() == input.size()
    assert logdet.size() == (batch_size,)
    assert z.size() == input.size()
    assert z_logdet.size() == (batch_size,)
    allclose(output, input)
    allclose(logdet, zeros)

    # w/ 2D padding mask, but all False (identical to w/o padding mask)
    kernel_size = 3
    dilation_rate = 2  # default of WaveNetAffineCoupling

    masked_model = MaskedWaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
    )
    non_masked_model = WaveNetAffineCoupling(
        coupling_channels,
        hidden_channels,
        num_layers=num_layers,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
    )

    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(masked_model.coupling.bottleneck_conv1d_out.bias.data)

    for p_masked, p_non_masked in zip(
        masked_model.coupling.parameters(), non_masked_model.coupling.parameters()
    ):
        p_non_masked.data.copy_(p_masked.data)

    input = torch.randn(batch_size, 2 * coupling_channels, max_length)
    padding_mask = torch.full((batch_size, max_length), fill_value=False)

    masked_z = masked_model(input, padding_mask=padding_mask)
    masked_output = masked_model(masked_z, padding_mask=padding_mask, reverse=True)
    non_masked_z = non_masked_model(input)
    non_masked_output = non_masked_model(non_masked_z, reverse=True)

    allclose(masked_z, non_masked_z)
    allclose(masked_output, non_masked_output)

    zeros = torch.zeros((batch_size,))

    masked_z, masked_z_logdet = masked_model(
        input,
        padding_mask=padding_mask,
        logdet=zeros,
    )
    masked_output, masked_logdet = masked_model(
        masked_z,
        padding_mask=padding_mask,
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

    allclose(masked_z, non_masked_z)
    allclose(masked_z_logdet, non_masked_z_logdet)
    allclose(masked_output, non_masked_output)
    allclose(masked_logdet, non_masked_logdet)


def test_stacked_residual_conv_block():
    torch.manual_seed(0)

    batch_size, max_length = 2, 16
    in_channels, hidden_channels = 4, 6
    num_layers = 3

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length).item()
    input = torch.randn((batch_size, in_channels, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    model = MaskedStackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
    )
    nn.init.normal_(model.bottleneck_conv1d_out.weight.data)
    nn.init.normal_(model.bottleneck_conv1d_out.bias.data)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    log_s, t = model(input, padding_mask=padding_mask)

    assert log_s.size() == (batch_size, in_channels, max_length)
    assert t.size() == (batch_size, in_channels, max_length)


def train_dummy_masked_act_norm1d(
    rank: int,
    world_size: int,
    port: int,
    num_features: int = 6,
    seed: int = 0,
    path: str = None,
) -> None:
    batch_size = 4
    max_length = 20

    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    num_threads = torch.get_num_threads()
    num_threads = max(num_threads // world_size, 1)
    torch.set_num_threads(num_threads)

    config = {
        "seed": seed,
        "distributed": {
            "enable": True,
            "backend": "gloo",
            "init_method": None,
        },
        "cudnn": {
            "benchmark": None,
            "deterministic": None,
        },
        "amp": {
            "enable": False,
            "accelerator": "cpu",
        },
    }

    config = OmegaConf.create(config)

    dist.init_process_group(backend=config.distributed.backend)
    torch.manual_seed(config.seed)

    g = torch.Generator()
    g.manual_seed(rank)

    model = MaskedActNorm1d(num_features)
    model = nn.parallel.DistributedDataParallel(model)

    length = torch.randint(1, max_length + 1, (batch_size,), dtype=torch.long)
    max_length = torch.max(length)
    input = torch.randn((batch_size, num_features, max_length))
    padding_mask = torch.arange(max_length) >= length.unsqueeze(dim=-1)

    input = input.masked_fill(padding_mask.unsqueeze(dim=1), 0)
    z = model(input, padding_mask=padding_mask)
    output = model(z, padding_mask=padding_mask, reverse=True)

    allclose(output, input)

    state_dict = {
        "latent": z,
        "length": length,
        "model": model.module.state_dict(),
    }
    torch.save(state_dict, path)

    dist.destroy_process_group()
