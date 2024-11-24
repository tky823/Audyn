"""Implementation of LEX (length-extrapolatable) Transformer.
Ported from https://github.com/pytorch/pytorch/blob/e8836759d0898c29262b5370e16970d697cbaf3a/torch/nn/modules/transformer.py.  # noqa: E501
"""

import warnings

from ..modules.lextransformer import LEXTransformerDecoder as _LEXTransformerDecoder
from ..modules.lextransformer import (
    LEXTransformerDecoderLayer as _LEXTransformerDecoderLayer,
)
from ..modules.lextransformer import LEXTransformerEncoder as _LEXTransformerEncoder
from ..modules.lextransformer import (
    LEXTransformerEncoderLayer as _LEXTransformerEncoderLayer,
)

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


class LEXTransformerDecoderLayer(_LEXTransformerDecoderLayer):
    """Decoder layer of LEX Transformer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )
