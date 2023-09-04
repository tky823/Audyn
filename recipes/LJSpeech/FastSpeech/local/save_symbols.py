import os

import torch
from omegaconf import DictConfig
from torchtext.vocab import build_vocab_from_iterator

import audyn
from audyn.utils.data.cmudict import BREAK_SYMBOLS, SPECIALS, symbols


@audyn.main()
def main(config: DictConfig) -> None:
    symbols_path = config.preprocess.symbols_path

    assert symbols_path is not None, "Specify preprocess.symbols_path."

    symbols_dir = os.path.dirname(symbols_path)
    os.makedirs(symbols_dir, exist_ok=True)

    vocab = build_vocab_from_iterator([symbols + BREAK_SYMBOLS], specials=SPECIALS)

    assert (
        len(vocab) == config.data.text.vocab_size
    ), f"Size of vocabulary is expected {config.data.text.vocab_size}, but given {len(vocab)}."

    torch.save(vocab, symbols_path)


if __name__ == "__main__":
    main()
