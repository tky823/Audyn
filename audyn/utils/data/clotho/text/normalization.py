import re

from ....text.normalization import BaseTextNormalizer

__all__ = ["ClothoTextNormalizer"]


class ClothoTextNormalizer(BaseTextNormalizer):
    def __init__(self) -> None:
        super().__init__()

        self.sub_re = re.compile(r'\s([,.!?;:"](?:\s|$))')
        self.continuous_space_re = re.compile(r"\s{2,}")

    def normalize(self, text: str) -> str:
        text = self.sub_re.sub(r"\1", text)
        text = self.continuous_space_re.sub(" ", text)

        return text
