import json
import os
import shutil
import uuid

from ..._github import download_file_from_github_release


def load_mammal_name_to_index(chunk_size: int = 8192) -> dict[str, int]:
    from ... import audyn_cache_dir

    wordnet_root = os.path.join(audyn_cache_dir, "data", "WordNet")

    url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/wordnet_mammal.json"

    if chunk_size is None:
        chunk_size = 8192

    if wordnet_root:
        os.makedirs(wordnet_root, exist_ok=True)

    filename = os.path.basename(url)
    path = os.path.join(wordnet_root, filename)

    if not os.path.exists(path):
        download_wordnet_taxonomy(url, path, chunk_size=chunk_size)

    with open(path) as f:
        taxonomy: list[dict[str, str]] = json.load(f)

    name_to_index = {}

    for index, sample in enumerate(taxonomy):
        name = sample["name"]
        name_to_index[name] = index

    return name_to_index


def download_wordnet_taxonomy(url: str, path: str, chunk_size: int = 8192) -> None:
    temp_path = path + str(uuid.uuid4())[:8]

    try:
        download_file_from_github_release(url, temp_path, chunk_size=chunk_size)
        shutil.move(temp_path, path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)

        raise e
