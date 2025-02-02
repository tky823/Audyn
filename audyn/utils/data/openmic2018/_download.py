import json
import os
from typing import Any, Dict, List, Optional

from ... import audyn_cache_dir
from ..download import download_file


def download_all_tags() -> List[str]:
    """Download all instrument tags."""
    tags = [
        "accordion",
        "banjo",
        "bass",
        "cello",
        "clarinet",
        "cymbals",
        "drums",
        "flute",
        "guitar",
        "mallet_percussion",
        "mandolin",
        "organ",
        "piano",
        "saxophone",
        "synthesizer",
        "trombone",
        "trumpet",
        "ukulele",
        "violin",
        "voice",
    ]

    return tags


def download_all_metadata(
    root: Optional[str] = None,
    force_download: bool = False,
    chunk_size: int = 1024,
) -> List[Dict[str, Any]]:
    url = "https://github.com/tky823/Audyn/releases/download/v0.0.5/openmic2018_metadata.jsonl"
    filename = os.path.basename(url)

    if root is None:
        root = os.path.join(audyn_cache_dir, "data", "openmic2018")

    path = os.path.join(root, filename)

    download_file(
        url,
        path,
        force_download=force_download,
        chunk_size=chunk_size,
    )

    metadata = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            sample = json.loads(line)
            metadata.append(sample)

    return metadata
