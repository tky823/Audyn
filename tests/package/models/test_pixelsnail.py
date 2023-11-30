import torch
import torch.nn as nn

from audyn.models.pixelsnail import DiscretePixelSNAIL, PixelSNAIL


def test_pixelsnail() -> None:
    batch_size = 2
    in_channels, out_channels, hidden_channels = 3, 5, 4
    height, width = 5, 7
    kernel_size = 3
    num_heads = 4
    num_blocks, num_repeats = 3, 2
    kdim, vdim = 8, 16

    model = PixelSNAIL(
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_repeats=num_repeats,
        kdim=kdim,
        vdim=vdim,
    )
    input = torch.randn((batch_size, in_channels, height, width))
    output = model(input)

    assert output.size() == (batch_size, out_channels, height, width)


def test_categorical_pixelsnail() -> None:
    batch_size = 2
    num_classes, in_channels, hidden_channels = 3, 5, 4
    height, width = 5, 7
    kernel_size = 3
    num_heads = 4
    num_blocks, num_repeats = 3, 2
    kdim, vdim = 8, 16
    distribution = "categorical"

    embedding = nn.Embedding(num_classes, in_channels)
    backbone = PixelSNAIL(
        in_channels,
        num_classes,
        hidden_channels,
        kernel_size,
        num_heads=num_heads,
        num_blocks=num_blocks,
        num_repeats=num_repeats,
        kdim=kdim,
        vdim=vdim,
    )
    model = DiscretePixelSNAIL(embedding, backbone, distribution=distribution)
    input = torch.randint(0, num_classes, (batch_size, height, width), dtype=torch.long)
    output = model(input)

    assert output.size() == (batch_size, num_classes, height, width)

    initial_state = torch.randint(0, num_classes, (batch_size, 1, 1), dtype=torch.long)
    output = model.inference(initial_state, height=height, width=width)

    assert output.size() == (batch_size, height, width)
