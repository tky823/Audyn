import json
import os
import shutil
import uuid
from typing import Optional

from audyn.utils import audyn_cache_dir
from audyn.utils._download import DEFAULT_CHUNK_SIZE
from audyn.utils._github import download_file_from_github_release


def download_audioset_taxonomy(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[dict[str, str]]:
    url = "https://github.com/tky823/hyperaudioset/releases/download/v0.0.0/audioset.json"

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "audioset")

    path = os.path.join(root, "audioset-full-taxonomy.json")

    if not os.path.exists(path) or force_download:
        temp_path = path + str(uuid.uuid4())[:8]

        try:
            download_file_from_github_release(
                url, temp_path, force_download=force_download, chunk_size=chunk_size
            )
            shutil.move(temp_path, path)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)

            raise e

    with open(path) as f:
        taxonomy: list[dict[str, str]] = json.load(f)

    return taxonomy


def download_audioset_tags(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[str]:
    taxonomy = download_audioset_taxonomy(
        root=root,
        force_download=force_download,
        chunk_size=chunk_size,
    )

    tags = []

    for sample in taxonomy:
        tag = sample["name"]
        tags.append(tag)

    return tags


def download_audioset_name_to_index(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, int]:
    tags = download_audioset_tags(root=root, force_download=force_download, chunk_size=chunk_size)

    name_to_index = {}

    for index, tag in enumerate(tags):
        name_to_index[tag] = index

    return name_to_index
