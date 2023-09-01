from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTrBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        hidden_channels: int,
        num_heads: int = 2,
        kernel_size: List[int] = [9, 1],
        dropout: float = 1e-1,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.mha = MultiheadAttentionBlock(
            d_model,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.ffn = ConvBlock(
            d_model, hidden_channels, kernel_size, dropout=dropout, **factory_kwargs
        )

        self.num_heads = num_heads
        self.batch_first = batch_first

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.BoolTensor] = None,
        src_key_padding_mask: Optional[torch.BoolTensor] = None,
        need_weights: Optional[torch.BoolTensor] = True,
        average_attn_weights: Optional[torch.BoolTensor] = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass of feed-forward transformer block for FastSpeech.

        Args:
            src (torch.Tensor): Source embedding of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise (src_length, batch_size, embed_dim).
            src_mask (torch.Tensor): Attention mask for source of shape
                (src_length, src_length) or (batch_size * num_heads, src_length, src_length).
            src_key_padding_mask (torch.Tensor): Padding mask of shape
                (src_length,) or (batch_size, src_length).
            need_weights (bool): If ``True``, this method returns ``output`` and ``attn_weights``.
                Otherwise, only ``output`` is returned.
            average_attn_weights (bool): If ``True``, attention mask is averaged over heads
                and the shape is (batch_size, src_length, src_length). Otherwise,
                (batch_size, num_heads, src_length, src_length).

        Returns:
            torch.Tensor: Output of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise, (src_length, batch_size, embed_dim).
            torch.Tensor: Attention weights of shape (batch_size, src_length, src_length)
                if ``average_attn_weights=True``. Otherwise,
                (batch_size, num_heads, src_length, src_length).
        """
        batch_first = self.batch_first

        attn_output, attn_weights = self.mha(
            src,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            attn_mask=src_mask,
            average_attn_weights=average_attn_weights,
        )

        if batch_first:
            attn_output = attn_output.permute(0, 2, 1)
        else:
            attn_output = attn_output.permute(1, 2, 0)

        if src_key_padding_mask is not None:
            # Since transformation of self.ffn is not position-wise operation,
            # padding mask is applied to attn_output here.
            if src_key_padding_mask.dim() == 1:
                padding_mask = src_key_padding_mask.unsqueeze(dim=0)
            elif src_key_padding_mask.dim() == 2:
                padding_mask = src_key_padding_mask
            else:
                raise ValueError(
                    "src_key_padding_mask is expected to be 1 or 2D, but {}D is given."
                )

            padding_mask = padding_mask.unsqueeze(dim=1)
            attn_output = attn_output.masked_fill(padding_mask, 0)

        x = self.ffn(attn_output, padding_mask=src_key_padding_mask)

        if batch_first:
            output = x.permute(0, 2, 1)
        else:
            output = x.permute(2, 0, 1)

        if need_weights:
            return output, attn_weights

        return output


class ConvBlock(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_channels: int,
        kernel_size: Union[List[int], int],
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        if type(kernel_size) is int:
            kernel_size = [kernel_size, kernel_size]

        k1, k2 = kernel_size

        assert k1 % 2 == 1 and k2 % 2 == 1, "Kernel sizes should be odd."

        self.conv1d_1 = nn.Conv1d(
            num_features, hidden_channels, kernel_size=k1, stride=1, **factory_kwargs
        )
        self.activation = nn.ReLU()
        self.conv1d_2 = nn.Conv1d(
            hidden_channels,
            num_features,
            kernel_size=k2,
            stride=1,
            bias=False,
            **factory_kwargs,
        )
        self.layer_norm = nn.LayerNorm(num_features, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

        self.kernel_size = kernel_size

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Forward pass of ConvBlock.

        Args:
            input (torch.Tensor): Input feature of shape (batch_size, num_features, length).
            padding_mask (torch.Tensor): Padding mask of shape (src_length,)
                or (batch_size, src_length).

        Returns:
            torch.Tensor: Output feature of shape (batch_size, num_features, length).

        """
        k1, k2 = self.kernel_size

        if padding_mask is not None:
            if padding_mask.dim() == 2:
                padding_mask = padding_mask.unsqueeze(dim=1)
            elif padding_mask.dim() != 1:
                raise ValueError(
                    f"padding_mask is expected to be 1 or 2D, but {padding_mask.dim()}D is given."
                )

        residual = input

        padding_left = (k1 - 1) // 2
        padding_right = k1 - 1 - padding_left
        x = F.pad(input, (padding_left, padding_right))

        x = self.conv1d_1(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0)

        x = self.activation(x)

        padding_left = (k2 - 1) // 2
        padding_right = k2 - 1 - padding_left
        x = F.pad(x, (padding_left, padding_right))

        x = self.conv1d_2(x)

        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0)

        x = self.dropout(x)
        x = x + residual
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        output = x.permute(0, 2, 1)

        return output


class MultiheadAttentionBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        factory_kwargs = {
            "device": device,
            "dtype": dtype,
        }

        self.mha = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            bias=bias,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            kdim=kdim,
            vdim=vdim,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        key_padding_mask: Optional[torch.BoolTensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.BoolTensor] = None,
        average_attn_weights: bool = True,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Forward pass of MultiheadAttentionBlock.

        Args:
            src (torch.Tensor): Source embedding of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise (src_length, batch_size, embed_dim).
            key_padding_mask (torch.BoolTensor, optional): Padding mask of shape
                (src_length,) or (batch_size, src_length).
            need_weights (bool): If ``True``, this method returns ``output`` and ``attn_weights``.
                Otherwise, only ``output`` is returned.
            attn_mask (torch.BoolTensor, optional): Attention mask of shape
                (src_length, src_length) or (batch_size * num_heads, src_length, src_length).
            average_attn_weights (bool): If ``True``, attention mask is averaged over heads
                and the shape is (batch_size, src_length, src_length). Otherwise,
                (batch_size, num_heads, src_length, src_length).

        Returns:
            torch.Tensor: Output of shape (batch_size, src_length, embed_dim)
                if ``batch_first=True``. Otherwise, (src_length, batch_size, embed_dim).
            torch.Tensor: Attention weights of shape (batch_size, src_length, src_length)
                if ``average_attn_weights=True``. Otherwise,
                (batch_size, num_heads, src_length, src_length).

        """
        residual = src

        attn_output, attn_weights = self.mha(
            src,
            src,
            src,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            attn_mask=attn_mask,
            average_attn_weights=average_attn_weights,
        )

        x = self.dropout(attn_output)
        output = self.layer_norm(x + residual)

        if need_weights:
            return output, attn_weights

        return output
