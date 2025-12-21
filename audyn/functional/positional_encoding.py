import torch


def rotary_positional_encoding(
    input: torch.Tensor, base: float = 10000, batch_first: bool = True
) -> torch.Tensor:
    """Rotary positional encoding (RoPE).

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
