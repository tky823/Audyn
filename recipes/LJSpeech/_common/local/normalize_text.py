import os
from concurrent.futures import ProcessPoolExecutor
from typing import Tuple

from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import instantiate
from audyn.utils.text import TextPreprocessor


@audyn.main()
def main(config: DictConfig) -> None:
    metadata_path = config.preprocess.metadata_path
    text_dir = config.preprocess.text_dir
    max_workers = config.preprocess.max_workers

    text_preprocessor = instantiate(config.data.text.preprocessor)

    with open(metadata_path) as f:
        data = []

        for line in f:
            line = line.strip()
            filename, text = parse_text(line)

            sample = {
                "filename": filename,
                "text": text,
            }
            data.append(sample)

    if max_workers > 1:
        for sample in data:
            filename = sample["filename"]
            text = sample["text"]
            text_path = os.path.join(text_dir, f"{filename}.txt")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                future = executor.submit(
                    process,
                    source_text=text,
                    text_path=text_path,
                    text_preprocessor=text_preprocessor,
                )
                futures.append(future)

        for future in tqdm(futures):
            future.result()
    else:
        for sample in tqdm(data):
            filename = sample["filename"]
            text = sample["text"]
            text_path = os.path.join(text_dir, f"{filename}.txt")

            process(
                source_text=text,
                text_path=text_path,
                text_preprocessor=text_preprocessor,
            )


def process(
    source_text: str,
    text_path: str,
    text_preprocessor: TextPreprocessor,
) -> None:
    save_dir = os.path.dirname(text_path)

    os.makedirs(save_dir, exist_ok=True)

    normalized_text = text_preprocessor.normalize_text(source_text)

    with open(text_path, mode="w") as f:
        f.write(normalized_text + "\n")


def parse_text(raw_text: str) -> Tuple[str, str]:
    """Parse text of LJSpeech.

    .. examples:

        >>> text = "LJ001-0045|1469, 1470;|fourteen sixty-nine, fourteen seventy;"
        >>> parse_text(text)
        ('LJ001-0045', 'fourteen sixty-nine, fourteen seventy;')
        >>> text = "LJ001-0002|in being comparatively modern.|in being comparatively modern."
        >>> parse_text(text)
        ('LJ001-0002', 'in being comparatively modern.')

    """
    filename, _, text = raw_text.split("|")

    return filename, text


if __name__ == "__main__":
    main()
