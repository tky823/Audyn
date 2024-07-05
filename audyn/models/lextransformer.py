"""Implementation of LEX (length-extrapolatable) Transformer.
Ported from https://github.com/pytorch/pytorch/blob/e8836759d0898c29262b5370e16970d697cbaf3a/torch/nn/modules/transformer.py.  # noqa: E501
"""

import warnings

from ..modules.lextransformer import LEXTransformerDecoder as _LEXTransformerDecoder
from ..modules.lextransformer import LEXTransformerDecoderLayer as _LEXTransformerDecoderLayer
from ..modules.lextransformer import LEXTransformerEncoder as _LEXTransformerEncoder
from ..modules.lextransformer import LEXTransformerEncoderLayer as _LEXTransformerEncoderLayer

__all__ = [
    "LEXTransformerEncoder",
    "LEXTransformerDecoder",
    "LEXTransformerEncoderLayer",
    "LEXTransformerDecoderLayer",
]

warning_message = (
    "audyn.modeles.lextransformer.{class_name} is deprecated."
    " Use audyn.modules.lextransformer.{class_name} instead."
)


class LEXTransformerEncoder(_LEXTransformerDecoder):
    """Encoder of LEX Transformer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


class LEXTransformerDecoder(_LEXTransformerEncoder):
    """Decoder of LEX Transformer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


class LEXTransformerEncoderLayer(_LEXTransformerEncoderLayer):
    """Encoder layer of LEX Transformer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


<<<<<<< HEAD
        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        if activation is F.relu or isinstance(activation, nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0

        self.activation = activation

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (torch.Tensor): the sequence to the encoder layer.
            src_mask (torch.BoolTensor, optional): the mask for the src sequence.
            src_key_padding_mask (torch.BoolTensor, optional): the mask for the src keys
                per batch.
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout2(x)


class LEXTransformerDecoderLayer(nn.Module):
=======
class LEXTransformerDecoderLayer(_LEXTransformerDecoderLayer):
>>>>>>> 940036f1 ([src] Create audyn.modules.lextransformer)
    """Decoder layer of LEX Transformer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )
<<<<<<< HEAD
        self.multihead_attn = ExtrapolatablePositionalMultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=batch_first,
            bias=bias,
            **factory_kwargs,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.norm3 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, **layer_norm_kwargs, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = get_activation(activation)

        self.activation = activation

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> torch.Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt (torch.Tensor): the sequence to the decoder layer.
            memory (torch.Tensor): the sequence from the last layer of the encoder.
            tgt_mask (torch.BoolTensor, optional): the mask for the tgt sequence.
            memory_mask (torch.BoolTensor, optional): the mask for the memory sequence.
            tgt_key_padding_mask (torch.BoolTensor, optional): the mask for the tgt keys per batch.
            memory_key_padding_mask (torch.BoolTensor, optional): the mask for the memory keys
                per batch.
            tgt_is_causal: If specified, applies a causal mask as ``tgt mask``.
                Default: ``False``.
                Warning:
                ``tgt_is_causal`` provides a hint that ``tgt_mask`` is
                the causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.
            memory_is_causal: If specified, applies a causal mask as
                ``memory mask``.
                Default: ``False``.
                Warning:
                ``memory_is_causal`` provides a hint that
                ``memory_mask`` is the causal mask. Providing incorrect
                hints can result in incorrect execution, including
                forward and backward compatibility.

        """
        x = tgt

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal
            )
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                is_causal=memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, is_causal=tgt_is_causal)
            )
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    is_causal=memory_is_causal,
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if is_causal:
            raise NotImplementedError("is_causal=True is not supported.")

        x, _ = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))

        return self.dropout3(x)
=======
>>>>>>> 940036f1 ([src] Create audyn.modules.lextransformer)
