import csv
import os
from typing import List

import torch
from omegaconf import DictConfig
from torchtext.vocab import build_vocab_from_iterator

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    captions_path = config.preprocess.captions_path
    symbols_path = config.preprocess.symbols_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert captions_path is not None, "Specify preprocess.captions_path."
    assert symbols_path is not None, "Specify preprocess.symbols_path."

    symbols_dir = os.path.dirname(symbols_path)

    if symbols_dir:
        os.makedirs(symbols_dir, exist_ok=True)

    vocab = build_vocab_from_iterator(process(captions_path), specials=["<BOS>", "<EOS>"])

    torch.save(vocab, symbols_path)


def process(captions_path: str) -> List[str]:
    with open(captions_path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            _, *captions = line

            for caption in captions:
                tokens = [token.lower() for token in caption.split(" ")]

                yield tokens


if __name__ == "__main__":
    main()
