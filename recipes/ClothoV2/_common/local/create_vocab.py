import csv
import os
from typing import List

from omegaconf import DictConfig
from torchtext.vocab import build_vocab_from_iterator

import audyn
from audyn.utils.data.clotho import ClothoTextNormalizer, ClothoTextTokenizer


@audyn.main()
def main(config: DictConfig) -> None:
    captions_path = config.preprocess.captions_path
    vocab_path = config.preprocess.vocab_path

    assert captions_path is not None, "Specify preprocess.captions_path."
    assert vocab_path is not None, "Specify preprocess.vocab_path."

    normalizer = ClothoTextNormalizer()
    tokenizer = ClothoTextTokenizer()

    vocab_dir = os.path.dirname(vocab_path)

    if vocab_dir:
        os.makedirs(vocab_dir, exist_ok=True)

    vocab = build_vocab_from_iterator(
        process(captions_path, normalizer, tokenizer), specials=["<BOS>", "<EOS>"]
    )

    with open(vocab_path, mode="w") as f:
        for word in vocab.get_itos():
            f.write(word + "\n")


def process(
    captions_path: str, normalizer: ClothoTextNormalizer, tokenizer: ClothoTextTokenizer
) -> List[str]:
    with open(captions_path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            _, *captions = line

            for caption in captions:
                normalized_text = normalizer(caption)
                tokens = tokenizer(normalized_text)

                yield tokens


if __name__ == "__main__":
    main()
