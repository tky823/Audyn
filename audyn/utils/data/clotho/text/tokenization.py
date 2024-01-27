import re
from typing import List

from ....text.tokenization import BaseTextTokenizer

__all__ = ["ClothoTextTokenizer"]


class ClothoTextTokenizer(BaseTextTokenizer):
    def __init__(self) -> None:
        super().__init__()

        self.sub_re = re.compile('[,.!?;:"]')

    def tokenize(self, text: str) -> List[str]:
        tokens = self.sub_re.sub("", text)

        return tokens
