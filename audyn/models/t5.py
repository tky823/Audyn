"""T5 encoder models and Transformer-compatible encoder layers.

This module implements the encoder-side computation used by T5 while
reusing PyTorch's ``TransformerEncoder`` stack. T5 differs from the vanilla
PyTorch Transformer mainly in its RMS normalization, gated feed-forward
network, and relative position bias. The relative position bias is
computed once before the encoder stack and explicitly passed to every layer.
"""

import math
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modules.t5 import T5RelativePositionalMultiheadAttention
from ..modules.transformer import get_activation

__all__ = [
    "T5TransformerEncoder",
    "T5EncoderLayer",
    "T5Encoder",
]


class T5Encoder(nn.Module):
    """T5 encoder producing contextualized token representations.

    The encoder consists of an embedding layer followed by a
    :class:`T5TransformerEncoder`. A single relative attention bias table is
    shared by all encoder layers, as in T5. The table is stored as a
    parameter of this module rather than inside each attention layer.

    Args:
        embedding: Token embedding module. Its output dimension must match
            the ``d_model`` of the layers in ``backbone``.
        backbone: Transformer encoder stack. Its layers must expose
            ``self_attn.batch_first`` and ``self_attn.num_heads``.
        batch_first: If ``True``, ``input`` has shape
            ``(batch_size, length)`` and the feature tensors use
            ``(batch_size, length, embedding_dim)``. If ``False``,
            ``input`` has shape ``(length, batch_size)`` and the feature
            tensors use ``(length, batch_size, embedding_dim)``.
            This value must agree with the ``batch_first`` value of the
            backbone layers.
        num_relative_position_buckets: Even number of signed relative
            position buckets used by the bucketization scheme. The current
            parameter table has ``num_relative_position_buckets - 1`` rows
            because this implementation uses a custom zero-centered index
            layout. At least four buckets are required.
        relative_position_max_distance: Maximum distance used by the
            logarithmic buckets. Larger distances are mapped to the last
            bucket on their respective side.
        relative_position_bidirectional: Whether both relative directions are
            represented. Only ``True`` is currently supported.
        device: Optional device for the relative attention bias parameter.
        dtype: Optional dtype for the relative attention bias parameter.

    Shape:
        ``input`` is ``(batch_size, length)`` when ``batch_first=True`` and
        ``(length, batch_size)`` otherwise. ``padding_mask`` is always
        ``(batch_size, length)`` and uses ``True`` for padding positions.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        backbone: nn.TransformerEncoder,
        batch_first: bool = True,
        num_relative_position_buckets: int = 32,
        relative_position_max_distance: int = 128,
        relative_position_bidirectional: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.embedding = embedding
        self.backbone = backbone
        self.batch_first = batch_first

        if len(self.backbone.layers) == 0:
            raise ValueError("backbone must contain at least one layer.")

        if self.batch_first != self.backbone.layers[0].self_attn.batch_first:
            raise ValueError(
                "T5Encoder.batch_first must match the batch_first setting of the backbone layer."
            )

        if not relative_position_bidirectional:
            raise ValueError("Only relative_position_bidirectional=True is supported.")

        num_heads = self.backbone.layers[0].self_attn.num_heads

        if num_relative_position_buckets <= 0 or relative_position_max_distance <= 0:
            raise ValueError(
                "num_relative_position_buckets and "
                "relative_position_max_distance must be positive."
            )

        if num_relative_position_buckets < 4 or num_relative_position_buckets % 2 != 0:
            raise ValueError("num_relative_position_buckets must be even and at least 4.")

        num_half_buckets = num_relative_position_buckets // 2
        max_exact = num_half_buckets // 2

        if relative_position_max_distance <= max_exact:
            raise ValueError("relative_position_max_distance must be greater than max_exact.")

        self.num_relative_position_buckets = num_relative_position_buckets
        self.num_embedding_buckets = num_relative_position_buckets - 1
        self.relative_position_max_distance = relative_position_max_distance
        self.relative_position_bidirectional = relative_position_bidirectional

        if device is None:
            device = embedding.weight.device

        if dtype is None:
            dtype = embedding.weight.dtype

        self.relative_attention_bias = nn.Parameter(
            torch.zeros(
                self.num_embedding_buckets,
                num_heads,
                device=device,
                dtype=dtype,
            )
        )

    def forward(
        self,
        input: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """Encode a batch of token indices.

        Args:
            input: Token indices of shape ``(batch_size, length)`` if
                ``batch_first=True`` and ``(length, batch_size)`` otherwise.
            padding_mask: Boolean mask of shape ``(batch_size, length)``.
                ``True`` indicates padding and ``False`` indicates a valid token.

        Returns:
            Contextualized features with shape
            ``(batch_size, length, embedding_dim)`` when
            ``batch_first=True`` and
            ``(length, batch_size, embedding_dim)`` otherwise.

        """
        x = self.embedding(input)

        if self.batch_first:
            batch_size, length, _ = x.size()
        else:
            length, batch_size, _ = x.size()

        if len(self.backbone.layers) == 0:
            positional_embedding = None
        else:
            positional_embedding = self._compute_positional_embedding(length, x.device).to(
                dtype=x.dtype
            )

            positional_embedding = positional_embedding.unsqueeze(dim=0)
            positional_embedding = positional_embedding.expand(batch_size, -1, -1, -1)
            positional_embedding = positional_embedding.reshape(
                batch_size * positional_embedding.size(1), length, length
            )

        x = self.backbone(
            x,
            positional_embedding=positional_embedding,
            src_key_padding_mask=padding_mask,
        )

        return x

    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        """Map signed relative distances to embedding-table indices.

        Distances near zero use exact buckets. Larger absolute distances use
        logarithmically spaced buckets and are clipped at
        ``relative_position_max_distance``. The returned indices are laid
        out symmetrically around zero: negative distances occupy the lower
        indices, zero is represented by the center index, and positive
        distances occupy the upper indices.

        Args:
            relative_position: Tensor of signed distances. In the encoder,
                a positive value means that the key is to the right of the
                query under the convention used by
                :meth:`_compute_positional_embedding`.

        Returns:
            A ``torch.long`` tensor with the same shape as
            ``relative_position`` and values in
            ``[0, num_relative_position_buckets - 1]``.
        """
        num_half_buckets = self.num_relative_position_buckets // 2
        max_exact = num_half_buckets // 2

        distance = torch.abs(relative_position)
        is_near = distance < max_exact
        clamped_distance = torch.clamp(distance.float(), min=1)
        log_relative_position = torch.log(clamped_distance / max_exact)
        log_max_position = math.log(self.relative_position_max_distance / max_exact)
        rescaled_distance = (
            (num_half_buckets - max_exact) * log_relative_position / log_max_position
        )
        rescaled_distance = rescaled_distance + max_exact
        rescaled_distance = rescaled_distance.to(torch.long)
        rescaled_distance = rescaled_distance.clamp(max=num_half_buckets - 1)
        bucket_indices = torch.where(is_near, distance, rescaled_distance)
        bucket_indices = torch.where(relative_position > 0, bucket_indices, -bucket_indices)

        return bucket_indices + num_half_buckets - 1

    def _compute_positional_embedding(self, length: int, device: torch.device) -> torch.Tensor:
        """Compute the shared T5 relative position bias for a sequence.

        Args:
            length: Query and key sequence length.
            device: Device on which the position indices and bias are
                computed.

        Returns:
            Relative position bias of shape ``(num_heads, length, length)``.
                The first dimension indexes attention heads, followed by
                query and key positions.
        """
        indices = torch.arange(length, device=device)
        relative_position = indices.view(1, -1) - indices.view(-1, 1)
        buckets = self._relative_position_bucket(relative_position)
        positional_embedding = F.embedding(buckets, self.relative_attention_bias)

        return positional_embedding.permute(2, 0, 1)


class T5TransformerEncoder(nn.TransformerEncoder):
    """TransformerEncoder stack that forwards one shared T5 position bias.

    Unlike :class:`torch.nn.TransformerEncoder`, this stack accepts
    ``positional_embedding`` and passes the same tensor to every layer. This
    preserves the T5 design in which the relative position bias is computed
    once and reused throughout the encoder.

    The stack itself still follows the ``TransformerEncoder`` convention: its
    layers are stored in ``self.layers`` and an optional final ``self.norm``
    is applied after all layers.
    """

    def forward(
        self,
        src: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        positional_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run all encoder layers while sharing the position bias.

        Args:
            src: Input features with the layout expected by the layers.
            mask: Optional attention mask forwarded as ``src_mask``.
            src_key_padding_mask: Optional ``(batch_size, length)`` padding mask.
            is_causal: Causal attention flag forwarded to each layer. T5
                encoder attention rejects ``True``.
            positional_embedding: Shared relative position bias.

        Returns:
            Encoder output with the same shape as ``src``.
        """
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                positional_embedding=positional_embedding,
                is_causal=is_causal,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class T5EncoderLayer(nn.Module):
    """T5 encoder layer compatible with ``TransformerEncoder``.

    The default layer is pre-norm and uses RMSNorm. Its feed-forward block is
    the gated T5 form

    ``linear3(activation(linear1(x)) * linear2(x))``.

    T5 uses pre-norm by default. Post-norm is available through
    ``norm_first=False`` for compatibility with the PyTorch Transformer API.

    Args:
        d_model: Feature dimension.
        nhead: Number of self-attention heads. ``d_model`` must be divisible
            by ``nhead``.
        dim_feedforward: Hidden dimension of each gated feed-forward branch.
        dropout: Dropout probability used after self-attention and the
            feed-forward projection.
        activation: Activation applied to the first feed-forward branch. A
            callable or a name accepted by :func:`get_activation` may be
            supplied.
        layer_norm_eps: Epsilon used by RMSNorm.
        batch_first: Whether input features use
            ``(batch_size, length, embedding_dim)`` rather than
            ``(length, batch_size, embedding_dim)``.
        norm_first: If ``True``, use T5-style pre-normalization. If ``False``,
            use post-normalization.
        device: Optional device for module parameters.
        dtype: Optional dtype for module parameters.

    Inputs:
        ``src`` has shape ``(batch_size, length, embedding_dim)`` when
        ``batch_first=True`` and
        ``(length, batch_size, embedding_dim)`` otherwise.
        ``src_key_padding_mask`` has shape ``(batch_size, length)``.
        ``positional_embedding`` is a relative attention bias with shape
        ``(num_heads, length, length)``,
        ``(batch_size * num_heads, length, length)``, or
        ``(batch_size, num_heads, length, length)``.

    Returns:
        The transformed tensor with the same shape and layout as ``src``.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.gelu,
        layer_norm_eps: float = 1e-6,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.self_attn = T5RelativePositionalMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        self.norm1 = _get_normalization(
            d_model,
            eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.norm2 = _get_normalization(
            d_model,
            eps=layer_norm_eps,
            device=device,
            dtype=dtype,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=False, device=device, dtype=dtype)
        self.linear2 = nn.Linear(d_model, dim_feedforward, bias=False, device=device, dtype=dtype)
        self.linear3 = nn.Linear(dim_feedforward, d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation
        self.norm_first = norm_first

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        positional_embedding: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Apply self-attention and the gated feed-forward block.

        Args:
            src: Input feature tensor.
            src_mask: Optional additive or boolean attention mask.
            src_key_padding_mask: Optional padding mask. ``True`` entries
                are ignored as keys.
            positional_embedding: Relative position bias shared by the
                encoder stack.
            is_causal: Causal attention is not supported by the T5 encoder.

        Returns:
            Layer output with the same shape as ``src``.
        """
        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                positional_embedding,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    positional_embedding,
                    is_causal=is_causal,
                )
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        positional_embedding: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """Compute the self-attention residual branch."""
        x, _ = self.self_attn(
            x,
            x,
            x,
            padding_mask=key_padding_mask,
            positional_embedding=positional_embedding,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )

        return self.dropout(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the gated feed-forward residual branch."""
        x = self.activation(self.linear1(x)) * self.linear2(x)
        x = self.linear3(x)

        return self.dropout(x)


class GELU(nn.Module):
    """Gaussian error linear unit used by the T5 ``gelu_new`` variant."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Apply the approximate GELU used by T5."""
        return (
            0.5
            * input
            * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * (input**3))))
        )


def _get_normalization(
    normalized_shape: int,
    eps: float,
    device=None,
    dtype=None,
) -> nn.Module:
    """Construct the preferred RMS normalization implementation.

    PyTorch's native :class:`torch.nn.RMSNorm` is used when available. An
    Audyn fallback is used on older PyTorch versions.

    Args:
        normalized_shape: Size of the final feature dimension.
        eps: Numerical stability constant.
        device: Optional parameter device.
        dtype: Optional parameter dtype.

    Returns:
        An RMSNorm module.
    """
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(
            normalized_shape,
            eps=eps,
            device=device,
            dtype=dtype,
        )

    from ..modules.normalization import RMSNorm

    return RMSNorm(
        normalized_shape,
        eps=eps,
        device=device,
        dtype=dtype,
    )
