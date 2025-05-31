import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List

import torch
from omegaconf import DictConfig

import audyn
from audyn.utils import setup_config

try:
    import yt_dlp
    from yt_dlp import YoutubeDL
except ImportError:
    raise ImportError("Please install yt_dlp.")


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    csv_path = config.preprocess.csv_path
    jsonl_path = config.preprocess.jsonl_path
    download_dir = config.preprocess.download_dir

    download(
        csv_path=csv_path,
        jsonl_path=jsonl_path,
        download_dir=download_dir,
    )


def download(csv_path: str, jsonl_path: str, download_dir: str) -> None:
    """Download audios by ytdlp."""
    jsonl_dir = os.path.dirname(jsonl_path)

    if jsonl_dir:
        os.makedirs(jsonl_dir, exist_ok=True)

    crawled_ids = set()
    videos = {}

    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                _id = data["id"]
                crawled_ids.add(_id)

    with open(csv_path) as f:
        for idx, line in enumerate(f):
            # ignore header lines
            if idx < 3:
                continue

            line = line.strip()
            ytid, start, end, tags = line.split(", ")
            start = float(start)
            end = float(end)
            _id = f"{ytid}_{int(start):03d}-{int(end):03d}"
            tags = tags[1:-1]
            tags = tags.split(",")
            videos[_id] = {
                "id": _id,
                "ytid": ytid,
                "start": start,
                "end": end,
                "tags": tags,
            }

    ids = sorted(list(videos.keys()))
    ids = shuffle_ids(ids)

    base_ydl_opts = {
        "format": "m4a/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "ignoreerrors": True,
    }

    with open(jsonl_path, mode="w") as f:
        for _id in ids:
            if _id in crawled_ids:
                continue

            video = videos[_id]
            ytid = video["ytid"]
            start = int(video["start"])
            end = int(video["end"])
            url = f"https://www.youtube.com/watch?v={ytid}"

            now = datetime.now()
            ydl_opts = copy.deepcopy(base_ydl_opts)
            ydl_opts["outtmpl"] = {
                "default": f"{download_dir}/{now.strftime("%Y%m%d%H")}/{ytid}_{start:03d}-{end:03d}.%(ext)s",  # noqa: E501
            }
            ydl_opts["download_ranges"] = Callback(video=video)

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is not None:
                    path = info["requested_downloads"][-1]["filepath"]
                    video["root"] = download_dir
                    video["path"] = os.path.relpath(path, download_dir)

                    line = json.dumps(video)
                    f.write(line + "\n")


class Callback:
    def __init__(self, video: Dict[str, Any]) -> None:
        self.video = video

    def __call__(self, info_dict: Dict[str, Any], ydl: YoutubeDL) -> Iterable[Dict[str, float]]:
        video = self.video
        start, end = video["start"], video["end"]

        yield {
            "start_time": start,
            "end_time": end,
        }


def shuffle_ids(ids: List[str]) -> List[str]:
    indices = torch.randperm(len(ids))
    indices = indices.tolist()
    ids = [ids[idx] for idx in indices]

    return ids


if __name__ == "__main__":
    main()
