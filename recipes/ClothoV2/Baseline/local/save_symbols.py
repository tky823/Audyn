import os
from typing import List

import torch
from omegaconf import DictConfig
from torchtext.vocab import build_vocab_from_iterator

import audyn
from audyn.utils.text.tokenization import BaseTextTokenizer


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    text_dir = config.preprocess.text_dir
    symbols_path = config.preprocess.symbols_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert text_dir is not None, "Specify preprocess.text_dir."
    assert symbols_path is not None, "Specify preprocess.symbols_path."

    tokenizer = audyn.utils.instantiate(config.data.text.tokenization)

    symbols_dir = os.path.dirname(symbols_path)

    if symbols_dir:
        os.makedirs(symbols_dir, exist_ok=True)

    vocab = build_vocab_from_iterator(
        process(list_path, text_dir, tokenizer), specials=["<BOS>", "<EOS>"]
    )

    torch.save(vocab, symbols_path)


def process(list_path: str, text_dir: str, tokenizer: BaseTextTokenizer) -> List[str]:
    with open(list_path) as f_list:
        for filename in f_list:
            filename = filename.strip("\n")

            text_path = os.path.join(text_dir, f"{filename}.txt")

            with open(text_path) as f_text:
                for caption in f_text:
                    tokens = tokenizer(caption)

                    yield tokens


if __name__ == "__main__":
    main()
