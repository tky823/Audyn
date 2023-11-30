import os

import torch
from omegaconf import DictConfig
from torchtext.vocab import build_vocab_from_iterator

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    category_list_path = config.preprocess.category_list_path
    category_path = config.preprocess.category_path

    assert category_path is not None, "Specify preprocess.category_path."

    categories_dir = os.path.dirname(category_path)
    os.makedirs(categories_dir, exist_ok=True)

    categories = []

    with open(category_list_path) as f:
        for line in f:
            category = line.strip()
            categories.append(category)

    category_to_id = build_vocab_from_iterator([categories], specials=["<UNK>"])

    assert len(category_to_id) == config.data.num_categories, (
        f"Number of categories is expected {config.data.num_categories},"
        f" but given {len(category_to_id)}."
    )

    torch.save(category_to_id, category_path)


if __name__ == "__main__":
    main()
