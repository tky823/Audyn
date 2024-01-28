import re
from typing import List

from ....text.tokenization import BaseTextTokenizer

__all__ = ["ClothoTextTokenizer"]


class ClothoTextTokenizer(BaseTextTokenizer):
    """Text tokenizer for Clotho dataset.

    This class is ported from
    https://github.com/audio-captioning/dcase-2020-baseline/blob/654e808924f7c0d4e6e3d90cdcdaf28c9165a0ce/tools/dataset_creation.py. # noqa: E501
    """

    def __init__(self) -> None:
        super().__init__()

        self.sub_re = re.compile('[,.!?;:"]')

    def tokenize(self, text: str) -> List[str]:
        text = self.sub_re.sub("", text)
        tokens = text.strip().split()

        return tokens
