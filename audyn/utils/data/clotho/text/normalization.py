import re

from ....text.normalization import BaseTextNormalizer

__all__ = ["ClothoTextNormalizer"]

_normalize_re = re.compile(r'\s([,.!?;:"](?:\s|$))')
_fix_re = re.compile(r'([,.!?;:"])([a-zA-Z])')
_whitespace_re = re.compile(r"\s+")
_punctuation_re = re.compile('[,.!?;:"]')


class ClothoTextNormalizer(BaseTextNormalizer):
    """Text normalizer for Clotho dataset.

    This class is ported from
    https://github.com/audio-captioning/dcase-2020-baseline/blob/654e808924f7c0d4e6e3d90cdcdaf28c9165a0ce/tools/dataset_creation.py. # noqa: E501
    """

    def normalize(self, text: str) -> str:
        text = text.lower()
        text = _normalize_re.sub(r"\1", text)
        text = _fix_re.sub(r"\1 \2", text)
        text = _whitespace_re.sub(" ", text)
        text = _punctuation_re.sub("", text)

        return text
