import json
import os
import shutil
import uuid
from typing import Any, Optional

from ..._github import download_file_from_github_release
from ..download import DEFAULT_CHUNK_SIZE


def download_mammal_name_to_index(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    taxonomy = _download_mammal_taxonomy(
        root=root, force_download=force_download, chunk_size=chunk_size
    )

    name_to_index = {}

    for index, sample in enumerate(taxonomy):
        name = sample["name"]
        name_to_index[name] = index

    return name_to_index


def download_mammal_taxonomy(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, dict[str, Any]]:
    mammal_taxonomy = {}

    _mammal_taxonomy = _download_mammal_taxonomy(
        root=root, force_download=force_download, chunk_size=chunk_size
    )

    for tag in _mammal_taxonomy:
        name = tag["name"]
        mammal_taxonomy[name] = {
            "parent": tag["parent"],
            "child": tag["child"],
        }

    return mammal_taxonomy


def download_mammal_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    taxonomy = _download_mammal_taxonomy(
        root=root, force_download=force_download, chunk_size=chunk_size
    )

    tags = []

    for tag in taxonomy:
        name = tag["name"]
        tags.append(name)

    return tags


def _download_mammal_taxonomy(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    from ... import audyn_cache_dir

    url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/wordnet_mammal.json"
    filename = os.path.basename(url)

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "wordnet")

    path = os.path.join(root, filename)
    temp_path = path + str(uuid.uuid4())[:8]

    if not os.path.exists(path) or force_download:
        try:
            download_file_from_github_release(
                url,
                temp_path,
                force_download=force_download,
                chunk_size=chunk_size,
            )
            shutil.move(temp_path, path)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            raise e

    with open(path) as f:
        taxonomy: list[dict[str, str]] = json.load(f)

    return taxonomy
