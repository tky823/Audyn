import pytest
import torch
import torch.nn.functional as F
from dummy import allclose

from audyn.modules.wavenet import GatedConv1d, ResidualConvBlock1d

parameters_is_causal = [True, False]
parameters_dual_head = [True, False]
parameters_dilation = [1, 4]

batch_size = 2
num_frames = 16


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_residual_conv_block1d(dilation: int, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 6

    input = torch.randn(batch_size, in_channels, num_frames)

    model = ResidualConvBlock1d(
        in_channels,
        hidden_channels,
        stride=1,
        dilation=dilation,
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
        model = ResidualConvBlock1d(
            in_channels,
            hidden_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        buffered_output = zero

        for _ in range(num_frames):
            output, _ = model(buffered_output)
            buffered_output = torch.cat([zero, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output, _ = model.incremental_forward(last_output)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output, atol=1e-7)

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_residual_conv_block1d_local(dilation: int, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 6
    local_dim = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    local_conditioning = torch.randn(batch_size, local_dim, num_frames)

    model = ResidualConvBlock1d(
        in_channels,
        hidden_channels,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        dual_head=dual_head,
        local_dim=local_dim,
    )

    output, skip = model(input, local_conditioning=local_conditioning)

    if dual_head:
        assert output.size() == (batch_size, in_channels, num_frames)
    else:
        assert output is None

    assert skip.size() == (batch_size, hidden_channels, num_frames)

    if is_causal:
        # incremental forward
        model = ResidualConvBlock1d(
            in_channels,
            hidden_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
            local_dim=local_dim,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        local_conditioning = torch.randn(batch_size, local_dim, num_frames)
        buffered_output = zero

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, : frame_idx + 1]
            output, _ = model(buffered_output, local_conditioning=h_local)
            buffered_output = torch.cat([zero, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, frame_idx : frame_idx + 1]
            last_output, _ = model.incremental_forward(last_output, local_conditioning=h_local)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output, atol=1e-6)

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
@pytest.mark.parametrize("dual_head", parameters_dual_head)
def test_residual_conv_block1d_global(dilation: int, is_causal: bool, dual_head: bool):
    torch.manual_seed(0)

    in_channels = 4
    hidden_channels = 6
    global_dim = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    global_conditioning = torch.randn(batch_size, global_dim)

    model = ResidualConvBlock1d(
        in_channels,
        hidden_channels,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        dual_head=dual_head,
        global_dim=global_dim,
    )

    output, skip = model(input, global_conditioning=global_conditioning)

    if dual_head:
        assert output.size() == (batch_size, in_channels, num_frames)
    else:
        assert output is None

    assert skip.size() == (batch_size, hidden_channels, num_frames)

    if is_causal:
        # incremental forward
        model = ResidualConvBlock1d(
            in_channels,
            hidden_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
            global_dim=global_dim,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        global_conditioning = torch.randn(batch_size, global_dim)
        buffered_output = zero

        for _ in range(num_frames):
            output, _ = model(buffered_output, global_conditioning=global_conditioning)
            buffered_output = torch.cat([zero, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output, _ = model.incremental_forward(
                last_output, global_conditioning=global_conditioning
            )
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output, atol=1e-6)

    model.remove_weight_norm_()


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_gated_conv1d(dilation: int, is_causal: bool):
    torch.manual_seed(0)

    in_channels, out_channels = 4, 8

    input = torch.randn(batch_size, in_channels, num_frames)

    model = GatedConv1d(
        in_channels,
        out_channels,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
    )

    input = _pad(
        input,
        kernel_size=model.kernel_size,
        dilation=model.dilation,
        is_causal=model.is_causal,
    )
    x_tanh_target = model.conv1d_tanh(input)
    x_sigmoid_target = model.conv1d_sigmoid(input)

    x_tanh_output, x_sigmoid_output = model._fused_conv1d(
        input,
        conv1d_tanh=model.conv1d_tanh,
        conv1d_sigmoid=model.conv1d_sigmoid,
        stride=model.stride,
        dilation=model.dilation,
    )

    allclose(x_tanh_output, x_tanh_target)
    allclose(x_sigmoid_output, x_sigmoid_target)

    if is_causal:
        # incremental forward
        model = GatedConv1d(
            in_channels,
            in_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        buffered_output = zero

        for _ in range(num_frames):
            output = model(buffered_output)
            buffered_output = torch.cat([zero, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for _ in range(num_frames):
            last_output = model.incremental_forward(last_output)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_gated_conv1d_local(dilation: int, is_causal: bool):
    torch.manual_seed(0)

    in_channels, out_channels = 4, 8
    local_dim = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    local_conditioning = torch.randn(batch_size, local_dim, num_frames)

    model = GatedConv1d(
        in_channels,
        out_channels,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        local_dim=local_dim,
    )

    input = _pad(
        input,
        kernel_size=model.kernel_size,
        dilation=model.dilation,
        is_causal=model.is_causal,
    )
    x_tanh_target = model.conv1d_tanh(input)
    x_sigmoid_target = model.conv1d_sigmoid(input)
    y_tanh_target = model.local_conv1d_tanh(local_conditioning)
    y_sigmoid_target = model.local_conv1d_sigmoid(local_conditioning)

    x_tanh_target = x_tanh_target + y_tanh_target
    x_sigmoid_target = x_sigmoid_target + y_sigmoid_target

    x_tanh_output, x_sigmoid_output = model._fused_conv1d(
        input,
        conv1d_tanh=model.conv1d_tanh,
        conv1d_sigmoid=model.conv1d_sigmoid,
        stride=model.stride,
        dilation=model.dilation,
    )
    y_tanh_output, y_sigmoid_output = model._fused_conv1d(
        local_conditioning,
        conv1d_tanh=model.local_conv1d_tanh,
        conv1d_sigmoid=model.local_conv1d_sigmoid,
    )

    x_tanh_output = x_tanh_output + y_tanh_output
    x_sigmoid_output = x_sigmoid_output + y_sigmoid_output

    allclose(x_tanh_output, x_tanh_target)
    allclose(x_sigmoid_output, x_sigmoid_target)

    if is_causal:
        # incremental forward
        model = GatedConv1d(
            in_channels,
            in_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
            local_dim=local_dim,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        local_conditioning = torch.randn(batch_size, local_dim, num_frames)
        buffered_output = zero

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, : frame_idx + 1]
            output = model(buffered_output, local_conditioning=h_local)
            buffered_output = torch.cat([zero, output], dim=-1)

        _, buffered_output = torch.split(buffered_output, [1, num_frames], dim=-1)

        incremental_buffered_output = torch.zeros((batch_size, in_channels, 0), dtype=torch.float)
        last_output = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)

        for frame_idx in range(num_frames):
            h_local = local_conditioning[:, :, frame_idx : frame_idx + 1]
            last_output = model.incremental_forward(last_output, local_conditioning=h_local)
            incremental_buffered_output = torch.cat(
                [incremental_buffered_output, last_output], dim=-1
            )

        allclose(buffered_output, incremental_buffered_output)


@pytest.mark.parametrize("dilation", parameters_dilation)
@pytest.mark.parametrize("is_causal", parameters_is_causal)
def test_gated_conv1d_global(dilation: int, is_causal: bool):
    torch.manual_seed(0)

    in_channels, out_channels = 4, 8
    global_dim = 3

    input = torch.randn(batch_size, in_channels, num_frames)
    global_conditioning = torch.randn(batch_size, global_dim)

    # global conditioning
    model = GatedConv1d(
        in_channels,
        out_channels,
        stride=1,
        dilation=dilation,
        is_causal=is_causal,
        global_dim=global_dim,
    )

    input = _pad(
        input,
        kernel_size=model.kernel_size,
        dilation=model.dilation,
        is_causal=model.is_causal,
    )

    if global_conditioning.dim() == 2:
        global_conditioning = global_conditioning.unsqueeze(dim=-1)

    x_tanh_target = model.conv1d_tanh(input)
    x_sigmoid_target = model.conv1d_sigmoid(input)
    y_tanh_target = model.global_conv1d_tanh(global_conditioning)
    y_sigmoid_target = model.global_conv1d_sigmoid(global_conditioning)

    x_tanh_target = x_tanh_target + y_tanh_target
    x_sigmoid_target = x_sigmoid_target + y_sigmoid_target

    x_tanh_output, x_sigmoid_output = model._fused_conv1d(
        input,
        conv1d_tanh=model.conv1d_tanh,
        conv1d_sigmoid=model.conv1d_sigmoid,
        stride=model.stride,
        dilation=model.dilation,
    )
    y_tanh_output, y_sigmoid_output = model._fused_conv1d(
        global_conditioning,
        conv1d_tanh=model.global_conv1d_tanh,
        conv1d_sigmoid=model.global_conv1d_sigmoid,
    )

    x_tanh_output = x_tanh_output + y_tanh_output
    x_sigmoid_output = x_sigmoid_output + y_sigmoid_output

    allclose(x_tanh_output, x_tanh_target)
    allclose(x_sigmoid_output, x_sigmoid_target)

    if is_causal:
        # incremental forward
        model = GatedConv1d(
            in_channels,
            in_channels,
            stride=1,
            dilation=dilation,
            is_causal=is_causal,
            global_dim=global_dim,
        )

        zero = torch.zeros((batch_size, in_channels, 1), dtype=torch.float)
        global_conditioning = torch.randn(batch_size, global_dim)
        buffered_output = zero

        for _ in range(num_frames):
            output = model(buffered_output, global_conditioning=global_conditioning)
            buffered_output = torch.cat([zero, output], dim=-1)

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


def _pad(
    input: torch.Tensor,
    kernel_size: int,
    dilation: int = 1,
    is_causal: bool = True,
) -> torch.Tensor:
    padding = (kernel_size - 1) * dilation

    if is_causal:
        padding_left = padding
        padding_right = 0
    else:
        padding_left = padding // 2
        padding_right = padding - padding_left

    output = F.pad(input, (padding_left, padding_right))

    return output
