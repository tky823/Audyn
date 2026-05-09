import torch


def rotary_positional_embedding(
    input: torch.Tensor, base: float = 10000, batch_first: bool = True
) -> torch.Tensor:
    """Rotary positional embedding (RoPE).

    Args:
        input (torch.Tensor): Query or key of shape (batch_size, length, num_features)
            if ``batch_first=True``, otherwise (length, batch_size, num_features).
        base (float): Base value for calculating the frequencies. Default is 10000.
        batch_first (bool): If True, the input and output tensors are in
            (batch_size, length, num_features) format. If False, they are in
            (length, batch_size, num_features) format. Default is True.

    Returns:
        torch.Tensor: Output with same shape as input.

    """
    device = input.device

    if batch_first:
        x_cos = input
    else:
        x_cos = input.transpose(1, 0)

    batch_size, length, num_features = x_cos.size()

    x_cos = x_cos.view(batch_size, length, num_features // 2, 2)
    x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
    x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

    pos_seq = torch.arange(length)
    num_seq = torch.arange(0, num_features, 2) / num_features
    theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

    sin = torch.sin(theta)
    cos = torch.cos(theta)
    sin = sin.to(device)
    cos = cos.to(device)

    x = x_sin * sin.unsqueeze(dim=-1) + x_cos * cos.unsqueeze(dim=-1)
    x = x.view(batch_size, length, num_features)

    if batch_first:
        output = x
    else:
        output = x.transpose(1, 0).contiguous()

    return output


def extrapolatable_rotary_positional_embedding(
    input: torch.Tensor,
    invert_decay: bool,
    smooth: float = 0.4,
    base: float = 10000,
    batch_first: bool = True,
) -> torch.Tensor:
    """Extrapolatable rotary positional embedding (xPos).

    Args:
        input (torch.Tensor): Query or key of shape (batch_size, length, num_features)
            if ``batch_first=True``, otherwise (length, batch_size, num_features).
        invert_decay (bool): If ``True``, decay is inverted.
        smooth (float): Smoothing factor.
        base (float): Base value for calculating the frequencies. Default is 10000.
        batch_first (bool): If True, the input and output tensors are in
            (batch_size, length, num_features) format. If False, they are in
            (length, batch_size, num_features) format. Default is True.

    Returns:
        torch.Tensor: Output with same shape as input.

    """
    device = input.device

    if batch_first:
        x_cos = input
    else:
        x_cos = input.transpose(1, 0)

    batch_size, length, num_features = x_cos.size()

    x_cos = x_cos.view(batch_size, length, num_features // 2, 2)
    x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
    x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

    pos_seq = torch.arange(length)
    num_seq = torch.arange(0, num_features, 2) / num_features
    theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

    if invert_decay:
        decay = (1 + smooth) / (num_seq + smooth)
    else:
        decay = (num_seq + smooth) / (1 + smooth)

    decay = decay ** pos_seq.unsqueeze(dim=-1)

    sin = decay * torch.sin(theta)
    cos = decay * torch.cos(theta)
    sin = sin.to(device)
    cos = cos.to(device)
    x = x_sin * sin.unsqueeze(dim=-1) + x_cos * cos.unsqueeze(dim=-1)
    x = x.view(batch_size, length, num_features)

    if batch_first:
        output = x
    else:
        output = x.transpose(1, 0).contiguous()

    return output


def partial_rotary_positional_embedding(
    input: torch.Tensor, base: float = 10000, fraction: float = 0.5, batch_first: bool = True
) -> torch.Tensor:
    """Partial rotary positional embedding (RoPE).

    Args:
        input (torch.Tensor): Query or key tensor. Shape is (batch_size, length,
            num_features) if ``batch_first=True``, otherwise
            (length, batch_size, num_features).
        base (float): Base value for calculating frequencies. Default is 10000.
        fraction (float): Proportion of features (0.0 to 1.0) to which
            rotary transformation is applied. Default is 0.5.
        batch_first (bool): If True, input and output tensors are in
            (batch_size, length, num_features) format. If False, they are in
            (length, batch_size, num_features) format. Default is True.

    Returns:
        torch.Tensor: Output tensor where only first ``int(fraction * num_features)``
            dimensions are rotated. Same shape as input.

    """
    device = input.device

    if batch_first:
        x = input
    else:
        x = input.transpose(1, 0)

    batch_size, length, num_features = x.size()
    num_rotary_features = int(fraction * num_features)

    x_cos, x_identity = torch.split(
        x, [num_rotary_features, num_features - num_rotary_features], dim=-1
    )

    x_cos = x_cos.view(batch_size, length, num_rotary_features // 2, 2)
    x_sin_pre, x_sin_post = torch.unbind(x_cos, dim=-1)
    x_sin = torch.stack([-x_sin_post, x_sin_pre], dim=-1)

    pos_seq = torch.arange(length, device=device)
    num_seq = torch.arange(0, num_rotary_features, 2, device=device) / num_rotary_features
    theta = pos_seq.unsqueeze(dim=-1) / (base**num_seq)

    sin = torch.sin(theta)
    cos = torch.cos(theta)
    sin = sin.to(device)
    cos = cos.to(device)

    x = x_sin * sin.unsqueeze(dim=-1) + x_cos * cos.unsqueeze(dim=-1)
    x = x.view(batch_size, length, num_rotary_features)
    x = torch.cat([x, x_identity], dim=-1)

    if batch_first:
        output = x
    else:
        output = x.transpose(1, 0).contiguous()

    return output
