import pytest
import torch
import torch.nn as nn
from dummy import allclose

from audyn.models.wavenet import (
    MultiSpeakerWaveNet,
    PostNet,
    StackedResidualConvBlock1d,
    Upsample,
    WaveNet,
)

parameters_dilated = [True, False]
parameters_is_causal = [True, False]
parameters_dual_head = [True, False]
parameters_dilation = [1, 4]

batch_size = 2
num_frames = 16


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_wavenet(dilated: bool, is_causal: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 4
    num_layers, num_stacks = 2, 3

    discrete_input = torch.randint(0, in_channels, (batch_size, num_frames))
    continuous_input = torch.randn(batch_size, in_channels, num_frames)

    model = WaveNet(
        in_channels,
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        num_stacks=num_stacks,
        dilated=dilated,
        is_causal=is_causal,
    )

    output = model(discrete_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    output = model(continuous_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    if is_causal:
        # incremental forward
        model = WaveNet(
            in_channels,
            in_channels,
            hidden_channels,
            num_layers=num_layers,
            num_stacks=num_stacks,
            dilated=dilated,
            is_causal=is_causal,
        )

        # discrete
        initial_input = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output)
            output = torch.argmax(output, dim=1)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.full(
            (batch_size, 0), fill_value=in_channels // 2, dtype=torch.long
        )
        last_output = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)

        for _ in range(num_frames):
            last_output = model.incremental_forward(last_output)
            last_output = torch.argmax(last_output, dim=1)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(initial_input, max_length=num_frames)

        assert output.size() == (batch_size, num_frames)

        model.clear_buffer()

        # continuous
        initial_input = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output = model.incremental_forward(last_output)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(initial_input, max_length=num_frames)

        assert output.size() == (batch_size, in_channels, num_frames)

        model.clear_buffer()

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_wavenet_local(dilated: bool, is_causal: bool):
    torch.manual_seed(0)

    in_channels = 4
    n_mels = 5
    hop_length = 4

    hidden_channels = 4
    num_layers, num_stacks = 2, 3

    discrete_input = torch.randint(0, in_channels, (batch_size, num_frames))
    continuous_input = torch.randn(batch_size, in_channels, num_frames)
    local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)

    upsample = nn.ConvTranspose1d(
        n_mels,
        n_mels,
        kernel_size=num_frames // hop_length,
        stride=num_frames // hop_length,
    )
    model = WaveNet(
        in_channels,
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        num_stacks=num_stacks,
        dilated=dilated,
        is_causal=is_causal,
        upsample=upsample,
        local_channels=n_mels,
    )

    output = model(discrete_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    output = model(continuous_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    if is_causal:
        # incremental forward
        upsample = nn.ConvTranspose1d(
            n_mels,
            n_mels,
            kernel_size=num_frames // hop_length,
            stride=num_frames // hop_length,
        )
        model = WaveNet(
            in_channels,
            in_channels,
            hidden_channels,
            num_layers=num_layers,
            num_stacks=num_stacks,
            dilated=dilated,
            is_causal=is_causal,
            upsample=upsample,
            local_channels=n_mels,
        )

        # discrete
        initial_input = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)
        local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output, local_conditioning=local_conditioning)
            output = torch.argmax(output, dim=1)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.full(
            (batch_size, 0), fill_value=in_channels // 2, dtype=torch.long
        )
        last_output = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output, local_conditioning=local_conditioning
            )
            last_output = torch.argmax(last_output, dim=1)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input, local_conditioning=local_conditioning, max_length=num_frames
        )

        assert output.size() == (batch_size, num_frames)

        model.clear_buffer()

        # continuous
        initial_input = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output, local_conditioning=local_conditioning)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output, local_conditioning=local_conditioning
            )
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input, local_conditioning=local_conditioning, max_length=num_frames
        )

        assert output.size() == (batch_size, in_channels, num_frames)

        model.clear_buffer()

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_wavenet_global(dilated: bool, is_causal: bool):
    torch.manual_seed(0)

    in_channels = 4
    embed_dim = 4

    hidden_channels = 4
    num_layers, num_stacks = 2, 3

    discrete_input = torch.randint(0, in_channels, (batch_size, num_frames))
    continuous_input = torch.randn(batch_size, in_channels, num_frames)
    global_conditioning = torch.randn(batch_size, embed_dim)

    model = WaveNet(
        in_channels,
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        num_stacks=num_stacks,
        dilated=dilated,
        is_causal=is_causal,
        global_channels=embed_dim,
    )

    output = model(discrete_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    output = model(continuous_input)
    assert output.size() == (batch_size, in_channels, num_frames)

    if is_causal:
        # incremental forward
        model = WaveNet(
            in_channels,
            in_channels,
            hidden_channels,
            num_layers=num_layers,
            num_stacks=num_stacks,
            dilated=dilated,
            is_causal=is_causal,
            global_channels=embed_dim,
        )

        # discrete
        initial_input = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)
        global_conditioning = torch.randn(batch_size, embed_dim)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output, global_conditioning=global_conditioning)
            output = torch.argmax(output, dim=1)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.full(
            (batch_size, 0), fill_value=in_channels // 2, dtype=torch.long
        )
        last_output = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output, global_conditioning=global_conditioning
            )
            last_output = torch.argmax(last_output, dim=1)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input, global_conditioning=global_conditioning, max_length=num_frames
        )

        assert output.size() == (batch_size, num_frames)

        model.clear_buffer()

        # continuous
        initial_input = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        global_conditioning = torch.randn(batch_size, embed_dim)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(buffered_output, global_conditioning=global_conditioning)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output, global_conditioning=global_conditioning
            )
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input, global_conditioning=global_conditioning, max_length=num_frames
        )

        assert output.size() == (batch_size, in_channels, num_frames)

        model.clear_buffer()

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_multispk_wavenet(dilated: bool, is_causal: bool):
    torch.manual_seed(0)

    in_channels = 4
    n_mels = 5
    hop_length = 4

    hidden_channels = 4
    num_layers, num_stacks = 2, 3

    num_speakers = 4
    padding_idx = 0
    embed_dim = 5

    discrete_input = torch.randint(0, in_channels, (batch_size, num_frames))
    continuous_input = torch.randn(batch_size, in_channels, num_frames)
    local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)
    speaker = torch.randint(0, num_speakers, (batch_size,), dtype=torch.long)

    upsample = nn.ConvTranspose1d(
        n_mels,
        n_mels,
        kernel_size=num_frames // hop_length,
        stride=num_frames // hop_length,
    )
    speaker_encoder = nn.Embedding(
        num_embeddings=num_speakers,
        embedding_dim=embed_dim,
        padding_idx=padding_idx,
    )
    model = MultiSpeakerWaveNet(
        in_channels,
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        num_stacks=num_stacks,
        dilated=dilated,
        is_causal=is_causal,
        upsample=upsample,
        speaker_encoder=speaker_encoder,
        local_channels=n_mels,
        global_channels=embed_dim,
    )

    output = model(
        discrete_input,
        local_conditioning=local_conditioning,
        speaker=speaker,
    )
    assert output.size() == (batch_size, in_channels, num_frames)

    output = model(
        continuous_input,
        local_conditioning=local_conditioning,
        speaker=speaker,
    )
    assert output.size() == (batch_size, in_channels, num_frames)

    if is_causal:
        # incremental forward
        upsample = nn.ConvTranspose1d(
            n_mels,
            n_mels,
            kernel_size=num_frames // hop_length,
            stride=num_frames // hop_length,
        )
        speaker_encoder = nn.Embedding(
            num_embeddings=num_speakers,
            embedding_dim=embed_dim,
            padding_idx=padding_idx,
        )
        model = MultiSpeakerWaveNet(
            in_channels,
            in_channels,
            hidden_channels,
            num_layers=num_layers,
            num_stacks=num_stacks,
            dilated=dilated,
            is_causal=is_causal,
            upsample=upsample,
            speaker_encoder=speaker_encoder,
            local_channels=n_mels,
            global_channels=embed_dim,
        )

        # discrete
        initial_input = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)
        local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)
        speaker = torch.randint(0, num_speakers, (batch_size,), dtype=torch.long)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(
                buffered_output,
                local_conditioning=local_conditioning,
                speaker=speaker,
            )
            output = torch.argmax(output, dim=1)
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.full(
            (batch_size, 0), fill_value=in_channels // 2, dtype=torch.long
        )
        last_output = torch.full((batch_size, 1), fill_value=in_channels // 2, dtype=torch.long)
        global_conditioning = model.speaker_encoder(speaker)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
            last_output = torch.argmax(last_output, dim=1)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input,
            local_conditioning=local_conditioning,
            speaker=speaker,
            max_length=num_frames,
        )

        assert output.size() == (batch_size, num_frames)

        model.clear_buffer()

        # continuous
        initial_input = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        local_conditioning = torch.randn(batch_size, n_mels, num_frames // hop_length)
        speaker = torch.randint(0, num_speakers, (batch_size,), dtype=torch.long)
        buffered_output = initial_input

        for _ in range(num_frames):
            output = model(
                buffered_output,
                local_conditioning=local_conditioning,
                speaker=speaker,
            )
            buffered_output = torch.cat([initial_input, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        global_conditioning = model.speaker_encoder(speaker)

        for _ in range(num_frames):
            last_output = model.incremental_forward(
                last_output,
                local_conditioning=local_conditioning,
                global_conditioning=global_conditioning,
            )
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)

        model.clear_buffer()

        output = model.inference(
            initial_input,
            local_conditioning=local_conditioning,
            speaker=speaker,
            max_length=num_frames,
        )

        assert output.size() == (batch_size, in_channels, num_frames)

        model.clear_buffer()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_stacked_residual_conv_block1d(dilated: bool, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 8
    num_layers = 3

    input = torch.randn(batch_size, in_channels, num_frames)

    model = StackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        dilated=dilated,
        is_causal=is_causal,
        dual_head=dual_head,
    )

    output, skip = model(input)

    if dual_head:
        assert output.size() == (batch_size, in_channels, num_frames)
    else:
        assert output is None

    assert skip.size() == (batch_size, hidden_channels, num_frames)

    if is_causal:
        # incremental forward
        model = StackedResidualConvBlock1d(
            in_channels,
            hidden_channels,
            skip_channels=in_channels,
            num_layers=num_layers,
            dilated=dilated,
            is_causal=is_causal,
            dual_head=dual_head,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        buffered_skip = zero

        for _ in range(num_frames):
            _, skip = model(buffered_skip)
            buffered_skip = torch.cat([zero, skip], dim=-1)

        _, buffered_skip = torch.split(buffered_skip, [1, num_frames], dim=-1)

        incremental_buffered_skip = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_skip = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            _, last_skip = model.incremental_forward(last_skip)
            incremental_buffered_skip = torch.cat([incremental_buffered_skip, last_skip], dim=-1)

        allclose(buffered_skip, incremental_buffered_skip, atol=1e-7)

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_stacked_residual_conv_block1d_local(dilated: bool, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 6
    local_channels = 3

    num_layers = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    local_conditioning = torch.randn(batch_size, local_channels, num_frames)

    model = StackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        dilated=dilated,
        is_causal=is_causal,
        dual_head=dual_head,
        local_channels=local_channels,
    )

    output, skip = model(input, local_conditioning=local_conditioning)

    if dual_head:
        assert output.size() == (batch_size, in_channels, num_frames)
    else:
        assert output is None

    assert skip.size() == (batch_size, hidden_channels, num_frames)

    if is_causal:
        # incremental forward
        model = StackedResidualConvBlock1d(
            in_channels,
            hidden_channels,
            skip_channels=in_channels,
            num_layers=num_layers,
            dilated=dilated,
            is_causal=is_causal,
            dual_head=dual_head,
            local_channels=local_channels,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        local_conditioning = torch.randn(batch_size, local_channels, num_frames)
        buffered_skip = zero

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, : frame_idx + 1]
            _, skip = model(buffered_skip, local_conditioning=h_local)
            buffered_skip = torch.cat([zero, skip], dim=-1)

        _, buffered_skip = torch.split(buffered_skip, [1, num_frames], dim=-1)

        incremental_buffered_skip = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_skip = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, frame_idx : frame_idx + 1]
            _, last_skip = model.incremental_forward(last_skip, local_conditioning=h_local)
            incremental_buffered_skip = torch.cat([incremental_buffered_skip, last_skip], dim=-1)

        allclose(buffered_skip, incremental_buffered_skip, atol=1e-7)

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilated", parameters_dilated)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_stacked_residual_conv_block1d_global(dilated: bool, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 6
    global_channels = 3

    num_layers = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    global_conditioning = torch.randn(batch_size, global_channels)

    model = StackedResidualConvBlock1d(
        in_channels,
        hidden_channels,
        num_layers=num_layers,
        dilated=dilated,
        is_causal=is_causal,
        dual_head=dual_head,
        global_channels=global_channels,
    )

    output, skip = model(input, global_conditioning=global_conditioning)

    if dual_head:
        assert output.size() == (batch_size, in_channels, num_frames)
    else:
        assert output is None

    assert skip.size() == (batch_size, hidden_channels, num_frames)

    if is_causal:
        # incremental forward
        model = StackedResidualConvBlock1d(
            in_channels,
            hidden_channels,
            skip_channels=in_channels,
            num_layers=num_layers,
            dilated=dilated,
            is_causal=is_causal,
            dual_head=dual_head,
            global_channels=global_channels,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        global_conditioning = torch.randn(batch_size, global_channels)
        buffered_skip = zero

        for _ in range(num_frames):
            _, skip = model(buffered_skip, global_conditioning=global_conditioning)
            buffered_skip = torch.cat([zero, skip], dim=-1)

        _, buffered_skip = torch.split(buffered_skip, [1, num_frames], dim=-1)

        incremental_buffered_skip = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_skip = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            _, last_skip = model.incremental_forward(
                last_skip, global_conditioning=global_conditioning
            )
            incremental_buffered_skip = torch.cat([incremental_buffered_skip, last_skip], dim=-1)

        allclose(buffered_skip, incremental_buffered_skip, atol=1e-7)

    model.remove_weight_norm_()


@pytest.mark.parametrize("out_channels", [8, 1])
def test_post_net(out_channels: int):
    in_channels = 4
    num_layers = 2

    input = torch.randn(batch_size, in_channels, num_frames)

    model = PostNet(in_channels, out_channels, num_layers=num_layers)

    output = model(input)

    assert output.size() == (batch_size, out_channels, num_frames)


def test_upsample():
    in_channels, out_channels = 2, 3
    kernel_size, stride = 4, 2

    input = torch.randn(batch_size, in_channels, num_frames)

    model = Upsample(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    output = model(input)

    assert output.size() == (batch_size, out_channels, num_frames * stride)
