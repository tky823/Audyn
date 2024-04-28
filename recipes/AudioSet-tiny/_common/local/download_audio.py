import json
import os
from typing import Any, Dict, Iterable, List

import torch
from omegaconf import DictConfig

import audyn

try:
    import yt_dlp
    from yt_dlp import YoutubeDL
except ImportError:
    raise ImportError("Please install yt_dlp.")


@audyn.main()
def main(config: DictConfig) -> None:
    torch.manual_seed(config.system.seed)

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

    crawled_ytids = set()
    videos = {}

    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                data = json.loads(line)
                ytid = data["ytid"]
                crawled_ytids.add(ytid)

    with open(csv_path) as f:
        for idx, line in enumerate(f):
            # ignore header lines
            if idx < 3:
                continue

            line = line.strip()
            ytid, start, end, tags = line.split(", ")
            tags = tags[1:-1]
            tags = tags.split(",")
            videos[ytid] = {
                "ytid": ytid,
                "start": float(start),
                "end": float(end),
                "tags": tags,
            }

            if idx > 10:
                # We use only a few samples.
                break

    ytids = sorted(list(videos.keys()))
    ytids = shuffle_ytids(ytids)

    ydl_opts = {
        "outtmpl": {
            "default": f"{download_dir}/%(epoch-3600>%Y%d%m%H)s/%(id)s.%(ext)s",
        },
        "format": "m4a/bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "download_ranges": Callback(videos=videos),
        "ignoreerrors": True,
    }

    with open(jsonl_path, mode="w") as f:
        for ytid in ytids:
            if ytid in crawled_ytids:
                continue

            video = videos[ytid]
            url = f"https://www.youtube.com/watch?v={ytid}"

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                if info is not None:
                    path = info["requested_downloads"][-1]["filepath"]
                    video["root"] = download_dir
                    video["path"] = os.path.relpath(path, download_dir)

                    line = json.dumps(video)
                    f.write(line + "\n")


class Callback:
    def __init__(self, videos: Dict[str, Any]) -> None:
        self.videos = videos

    def __call__(self, info_dict: Dict[str, Any], ydl: YoutubeDL) -> Iterable[Dict[str, float]]:
        ytid = info_dict["id"]
        video = self.videos[ytid]
        start, end = video["start"], video["end"]

        yield {
            "start_time": start,
            "end_time": end,
        }


def shuffle_ytids(ytids: List[str]) -> List[str]:
    indices = torch.randperm(len(ytids)).tolist()
    ytids = [ytids[idx] for idx in indices]

    return ytids


if __name__ == "__main__":
    main()
