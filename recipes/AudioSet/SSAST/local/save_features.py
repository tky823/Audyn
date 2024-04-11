import json
import os
from typing import Any, Dict

import torch
import torchaudio
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    jsonl_path = config.preprocess.jsonl_path
    download_dir = config.preprocess.download_dir

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert jsonl_path is not None, "Specify preprocess.jsonl_path."
    assert download_dir is not None, "Specify preprocess.download_dir."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    os.makedirs(feature_dir, exist_ok=True)
    template_path = os.path.join(feature_dir, "%d.tar")

    max_shard_size = config.preprocess.max_shard_size

    videos = {}

    with open(jsonl_path) as f:
        for line in f:
            video = json.loads(line)
            ytid = video["ytid"]
            videos[ytid] = video

    with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
        for line in tqdm(f):
            filename = line.strip()
            video = videos[filename]
            process_webdataset(
                sink,
                video=video,
                download_dir=download_dir,
            )


def process_webdataset(
    sink: wds.ShardWriter,
    video: Dict[str, Any],
    download_dir: str,
) -> None:
    feature = {}

    ytid = video["ytid"]
    tags = video["tags"]
    m4a_path = os.path.join(download_dir, video["path"])
    metadata = torchaudio.info(m4a_path)

    with open(m4a_path, mode="rb") as f:
        audio = f.read()

    feature["__key__"] = ytid
    feature["audio.m4a"] = audio
    feature["tags.json"] = tags
    feature["filename.txt"] = ytid
    feature["sample_rate.pth"] = torch.tensor(metadata.sample_rate, dtype=torch.long)

    sink.write(feature)


if __name__ == "__main__":
    main()
