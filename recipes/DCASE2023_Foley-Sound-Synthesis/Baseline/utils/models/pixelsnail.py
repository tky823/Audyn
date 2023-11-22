import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

from audyn.modules.pixelsnail import _generate_square_subsequent_mask

IS_TORCH_LT_2_1 = version.parse(torch.__version__) < version.parse("2.1")


class CausalAttention2d(nn.Module):
    """Cross attention with causality for 2D input.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Embedding dimension of values, which is equal to number of
            output channels. ``out_channels`` should be divisible by ``num_heads``.
        kdim (int): Embedding dimension of keys. ``kdim`` should be divisible by ``num_heads``.
        num_heads (int): Number of heads in attention.
        dropout (float): Dropout rate in attention. Default: ``0.0``.
        weight_regularization (str, optional): Weight regularization.

    """

    def __init__(
        self,
        embed_dim: int,
        qdim: int,
        kdim: int,
        num_heads: int,
        dropout: float = 0.0,
        weight_regularization: Optional[str] = "weight_norm",
    ) -> None:
        super().__init__()

        self.q_proj = nn.Linear(qdim, embed_dim)
        self.k_proj = nn.Linear(kdim, embed_dim)
        self.v_proj = nn.Linear(kdim, embed_dim)

        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) should be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.qdim, self.kdim = qdim, kdim
        self.num_heads = num_heads
        self.dropout = dropout

        if weight_regularization is not None:
            if weight_regularization == "weight_norm":
                self.weight_norm_()
            elif weight_regularization == "spectral_norm":
                self.spectral_norm_()
            else:
                raise ValueError(
                    "{}-based weight regularization is not supported.".format(
                        weight_regularization
                    )
                )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """Forward pass of CausalSelfAttention2d.

        Args:
            query (torch.Tensor): Query of shape (batch_size, qdim, height, width).
            key (torch.Tensor): Key of shape (batch_size, kdim, height, width).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, embed_dim, height, width).

        """
        embed_dim = self.embed_dim
        qdim, kdim = self.qdim, self.kdim
        num_heads = self.num_heads
        dropout = self.dropout
        batch_size, _, height, width = query.size()

        query = query.permute(0, 2, 3, 1).contiguous()
        query = query.view(batch_size, height * width, qdim)
        key = key.permute(0, 2, 3, 1).contiguous()
        key = key.view(batch_size, height * width, kdim)

        query = self.q_proj(query)
        value = self.v_proj(key)
        key = self.k_proj(key)
        query = query.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        key = key.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        value = value.view(batch_size, height * width, num_heads, embed_dim // num_heads)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attn_score = torch.matmul(query, key) / math.sqrt(embed_dim // num_heads)
        attn_mask = self.generate_square_subsequent_mask(
            height * width,
            device=attn_score.device,
            dtype=attn_score.dtype,
        )
        attn_score = attn_score + attn_mask
        attn_weights = F.softmax(attn_score, dim=-1)
        attn_weights = F.dropout(attn_weights, p=dropout, training=self.training)
        x = torch.matmul(attn_weights, value)
        x = x.permute(0, 1, 3, 2).contiguous()
        output = x.view(batch_size, embed_dim, height, width)

        return output

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> torch.BoolTensor:
        return _generate_square_subsequent_mask(sz, device=device, dtype=dtype)

    def weight_norm_(self) -> None:
        """Set weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            weight_norm_fn = nn.utils.weight_norm
        else:
            weight_norm_fn = nn.utils.parametrizations.weight_norm

        self.q_proj = weight_norm_fn(self.q_proj)
        self.k_proj = weight_norm_fn(self.k_proj)
        self.v_proj = weight_norm_fn(self.v_proj)

    def remove_weight_norm_(self) -> None:
        """Remove weight_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_weight_norm_fn = nn.utils.remove_weight_norm
            remove_weight_norm_args = ()
        else:
            remove_weight_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_weight_norm_args = ("weight",)

        self.q_proj = remove_weight_norm_fn(self.q_proj, *remove_weight_norm_args)
        self.k_proj = remove_weight_norm_fn(self.k_proj, *remove_weight_norm_args)
        self.v_proj = remove_weight_norm_fn(self.v_proj, *remove_weight_norm_args)

    def spectral_norm_(self) -> None:
        """Set spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            spectral_norm_fn = nn.utils.spectral_norm
        else:
            spectral_norm_fn = nn.utils.parametrizations.spectral_norm

        self.q_proj = spectral_norm_fn(self.q_proj)
        self.k_proj = spectral_norm_fn(self.k_proj)
        self.v_proj = spectral_norm_fn(self.v_proj)

    def remove_spectral_norm_(self) -> None:
        """Remove spectral_norm from conv2d."""
        if IS_TORCH_LT_2_1:
            remove_spectral_norm_fn = nn.utils.remove_spectral_norm
            remove_spectral_norm_args = ()
        else:
            remove_spectral_norm_fn = nn.utils.parametrize.remove_parametrizations
            remove_spectral_norm_args = ("weight",)

        self.q_proj = remove_spectral_norm_fn(self.q_proj, *remove_spectral_norm_args)
        self.k_proj = remove_spectral_norm_fn(self.k_proj, *remove_spectral_norm_args)
        self.v_proj = remove_spectral_norm_fn(self.v_proj, *remove_spectral_norm_args)
