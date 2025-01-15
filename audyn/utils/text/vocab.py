from typing import List


class Vocab(dict):
    # NOTE: temporary fix
    def __call__(self, text: List[str]) -> List[int]:
        indices = []

        for token in text:
            index = self[token]
            indices.append(index)

        return indices
