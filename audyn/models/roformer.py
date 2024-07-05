import warnings

from ..modules.roformer import RoFormerDecoder as _RoFormerDecoder
from ..modules.roformer import RoFormerDecoderLayer as _RoFormerDecoderLayer
from ..modules.roformer import RoFormerEncoder as _RoFormerEncoder
from ..modules.roformer import RoFormerEncoderLayer as _RoFormerEncoderLayer

# for backward compatibility
__all__ = [
    "RoFormerEncoder",
    "RoFormerDecoder",
    "RoFormerEncoderLayer",
    "RoFormerDecoderLayer",
]

warning_message = (
    "audyn.modeles.roformer.{class_name} is deprecated."
    " Use audyn.modules.roformer.{class_name} instead."
)


class RoFormerEncoder(_RoFormerEncoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


class RoFormerDecoder(_RoFormerDecoder):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


class RoFormerEncoderLayer(_RoFormerEncoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )


class RoFormerDecoderLayer(_RoFormerDecoderLayer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        warnings.warn(
            warning_message.format(class_name=self.__class__.__name__),
            DeprecationWarning,
            stacklevel=2,
        )
