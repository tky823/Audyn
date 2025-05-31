"""Save features of samples in AudioSet.
- audio.m4a: Raw audio.
- tags.json: List of tags.
- filename.txt: Filename.
- sample_rate.pth: Sampling rate.
"""

import json
import os
from multiprocessing import Process, Queue
from typing import Any, Dict, List

import torch
import torchaudio
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import setup_config


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    feature_dir = config.preprocess.feature_dir
    jsonl_path = config.preprocess.jsonl_path
    max_workers = config.preprocess.max_workers
    max_shard_size = config.preprocess.max_shard_size

    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert jsonl_path is not None, "Specify preprocess.jsonl_path."
    assert max_workers is not None, "Specify preprocess.max_workers."

    if dump_format != "webdataset":
        raise ValueError("Only webdataset is supported as dump_format.")

    os.makedirs(feature_dir, exist_ok=True)
    template_path = os.path.join(feature_dir, "%d.tar")

    videos = {}

    with open(jsonl_path) as f:
        for line in f:
            video = json.loads(line)
            _id = video["id"]
            videos[_id] = video

    video_subsets = [[] for _ in range(max_workers)]

    with open(list_path) as f:
        for idx, line in tqdm(enumerate(f)):
            filename = line.strip()
            _id = os.path.basename(filename)
            video = videos[_id]
            video_subsets[idx % max_workers].append(video)

    queue = Queue()

    # load
    loading_processes: List[Process] = []

    for videos in video_subsets:
        p = Process(
            target=process_webdataset,
            args=(queue,),
            kwargs={
                "videos": videos,
            },
        )
        loading_processes.append(p)

    # write
    writing_process = Process(
        target=write_to_shards,
        args=(queue,),
        kwargs={
            "num_workers": max_workers,
            "tar_path": template_path,
            "max_shard_size": max_shard_size,
        },
    )

    # start multiprocessing
    for p in loading_processes:
        p.start()

    writing_process.start()

    # finish multiprocessing
    for p in loading_processes:
        p.join()

    writing_process.join()


def process_webdataset(
    queue: Queue,
    videos: List[Dict[str, Any]] = None,
) -> None:
    if videos is not None:
        for video in videos:
            feature = {}

            _id = video["id"]
            tags = video["tags"]
            root = video["root"]
            m4a_path = os.path.join(root, video["path"])
            metadata = torchaudio.info(m4a_path)

            with open(m4a_path, mode="rb") as f:
                audio = f.read()

            feature["__key__"] = _id
            feature["audio.m4a"] = audio
            feature["tags.json"] = tags
            feature["filename.txt"] = _id
            feature["sample_rate.pth"] = torch.tensor(
                metadata.sample_rate,
                dtype=torch.long,
            )

            queue.put(feature)

    queue.put(None)


def write_to_shards(
    queue: Queue,
    num_workers: int = 1,
    tar_path: str = None,
    max_shard_size: int = 1,
) -> None:
    num_working_processes = num_workers

    if tar_path is None:
        raise ValueError("Specify tar_path.")

    with wds.ShardWriter(tar_path, maxsize=max_shard_size) as sink:
        while True:
            feature = queue.get()

            if feature is None:
                num_working_processes -= 1
            else:
                sink.write(feature)

            if num_working_processes == 0:
                break


if __name__ == "__main__":
    main()
