import csv
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List

from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils.text.normalization import BaseTextNormalizer


@audyn.main()
def main(config: DictConfig) -> None:
    captions_path = config.preprocess.captions_path
    text_dir = config.preprocess.text_dir
    max_workers = config.preprocess.max_workers

    assert captions_path is not None, "Specify preprocess.captions_path."
    assert text_dir is not None, "Specify preprocess.text_dir."

    normalizer = audyn.utils.instantiate(config.data.text.normalization)

    os.makedirs(text_dir, exist_ok=True)

    captions = {}

    with open(captions_path) as f:
        reader = csv.reader(f)

        for idx, line in enumerate(reader):
            if idx == 0:
                continue

            filename, *_captions = line
            filename, _ = os.path.splitext(filename)
            captions[filename] = _captions

    config.preprocess.max_workers = 1
    max_workers = config.preprocess.max_workers

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            for filename, _captions in tqdm(captions.items()):
                text_path = os.path.join(text_dir, f"{filename}.txt")

                future = executor.submit(
                    process,
                    captions=_captions,
                    text_path=text_path,
                    normalizer=normalizer,
                )
                futures.append(future)

            for future in tqdm(futures):
                future.result()
    else:
        for filename, _captions in tqdm(captions.items()):
            text_path = os.path.join(text_dir, f"{filename}.txt")

            process(
                captions=_captions,
                text_path=text_path,
                normalizer=normalizer,
            )


def process(
    captions: List[str],
    text_path: str,
    normalizer: BaseTextNormalizer,
) -> None:
    with open(text_path, mode="w") as f:
        for caption in captions:
            caption = normalizer(caption)
            f.write(caption + "\n")


if __name__ == "__main__":
    main()
