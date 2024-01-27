import re

from ....text.normalization import BaseTextNormalizer

__all__ = ["ClothoTextNormalizer"]


class ClothoTextNormalizer(BaseTextNormalizer):
    """Text normalizer for Clotho dataset.

    This class is ported from
    https://github.com/audio-captioning/dcase-2020-baseline/blob/654e808924f7c0d4e6e3d90cdcdaf28c9165a0ce/tools/dataset_creation.py. # noqa: E501
    """

    def __init__(self) -> None:
        super().__init__()

        self.sub_re = re.compile(r'\s([,.!?;:"](?:\s|$))')
        self.continuous_space_re = re.compile(r"\s{2,}")

    def normalize(self, text: str) -> str:
        text = self.sub_re.sub(r"\1", text)
        text = self.continuous_space_re.sub(" ", text)
        text = text.lower()

        return text
