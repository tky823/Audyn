import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict

import torch
import torchaudio
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import setup_config
from audyn.utils.data.mtg_jamendo import download_all_metadata


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    feature_dir = config.preprocess.feature_dir
    split = config.preprocess.split

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert split is not None, "Specify preprocess.split."

    os.makedirs(feature_dir, exist_ok=True)

    metadata = download_all_metadata(split=split)
    metadata_by_filename = {}
    filenames = []

    with open(list_path) as f:
        for line in f:
            filename = line.strip()
            filenames.append(filename)

    for _metadata in metadata:
        path = _metadata["path"]
        filename = path.replace(".mp3", "")

        if filename in filenames:
            metadata_by_filename[filename] = _metadata

    if dump_format == "torch":
        max_workers = config.preprocess.max_workers

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                with open(list_path) as f:
                    for line in f:
                        filename = line.strip()
                        mp3_path = os.path.join(wav_dir, f"{filename}.mp3")
                        feature_path = os.path.join(feature_dir, f"{filename}.pth")
                        _metadata = metadata_by_filename[filename]

                        future = executor.submit(
                            process_torch,
                            filename=filename,
                            mp3_path=mp3_path,
                            feature_path=feature_path,
                            metadata=_metadata,
                        )
                        futures.append(future)

                for future in tqdm(futures):
                    future.result()
        else:
            with open(list_path) as f:
                for line in tqdm(f):
                    filename = line.strip()
                    mp3_path = os.path.join(wav_dir, f"{filename}.mp3")
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")
                    _metadata = metadata_by_filename[filename]

                    process_torch(
                        filename=filename,
                        wav_path=mp3_path,
                        feature_path=feature_path,
                        metadata=_metadata,
                    )
    else:
        template_path = os.path.join(feature_dir, "%d.tar")

        max_shard_size = config.preprocess.max_shard_size

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                wav_path = os.path.join(wav_dir, f"{filename}.mp3")
                _metadata = metadata_by_filename[filename]

                process_webdataset(
                    sink,
                    filename=filename,
                    wav_path=wav_path,
                    metadata=_metadata,
                )


def process_torch(
    filename: str, wav_path: str, feature_path: str, metadata: Dict[str, Any]
) -> None:
    feature = {}

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)

    track = metadata["track"]
    artist = metadata["artist"]
    album = metadata["album"]
    duration = metadata["duration"]
    tags = metadata["tags"]

    feature["waveform"] = waveform
    feature["sample_rate"] = torch.tensor(sample_rate, dtype=torch.long)
    feature["track"] = track
    feature["artist"] = artist
    feature["album"] = album
    feature["duration"] = torch.tensor(duration, dtype=torch.float)
    feature["tags"] = tags
    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    sink: wds.ShardWriter, filename: str, wav_path: str, metadata: Dict[str, Any]
) -> None:
    feature = {}

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)

    track = metadata["track"]
    artist = metadata["artist"]
    album = metadata["album"]
    duration = metadata["duration"]
    tags = metadata["tags"]

    feature["__key__"] = filename
    feature["waveform.pth"] = waveform
    feature["sample_rate.pth"] = torch.tensor(sample_rate, dtype=torch.long)
    feature["track.txt"] = track
    feature["artist.txt"] = artist
    feature["album.txt"] = album
    feature["duration.pth"] = torch.tensor(duration, dtype=torch.float)
    feature["tags.json"] = tags
    feature["filename.txt"] = filename

    sink.write(feature)


if __name__ == "__main__":
    main()
