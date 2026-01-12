import json
import os
from typing import Any, Dict, List, Optional

from ... import audyn_cache_dir
from ..._github import download_file_from_github_release
from ..download import DEFAULT_CHUNK_SIZE


def download_metadata(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> List[Dict[str, Any]]:
    base_url = "https://github.com/tky823/Audyn/releases/download/v0.2.1/jamendo-max-caps.{index:02d}.jsonl"  # noqa: E501

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "jamendo-max-caps")

    metadata = []

    for index in range(4):
        url = base_url.format(index=index)
        filename = os.path.basename(url)
        path = os.path.join(root, filename)

        download_file_from_github_release(
            url,
            path,
            force_download=force_download,
            chunk_size=chunk_size,
        )

        _metadata = _load_metadata(path)
        metadata.extend(_metadata)

    return metadata


def _load_metadata(path: str) -> List[Dict[str, Any]]:
    metadata = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            data = json.loads(line)
            metadata.append(data)

    return metadata
