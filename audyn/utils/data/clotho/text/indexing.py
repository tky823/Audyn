from typing import List, Union

import torch
import torchtext

from ....text.indexing import BaseTextIndexer

__all__ = ["ClothoTextIndexer"]


class ClothoTextIndexer(BaseTextIndexer):
    def __init__(self, vocab: Union[str, torchtext.vocab.Vocab]) -> None:
        super().__init__()

        if isinstance(vocab, str):
            path = vocab
            vocab = torch.load(path)
        elif isinstance(vocab, torchtext.vocab.Vocab):
            pass
        else:
            raise NotImplementedError(f"{type(vocab)} is not supported.")

        self.vocab = vocab

    def index(self, text: List[str]) -> List[int]:
        tokens = self.vocab(text)

        return tokens
