import os

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
    fma_root = config.preprocess.fma_root

    assert dump_format is not None, "Specify preprocess.dump_format."
    assert list_path is not None, "Specify preprocess.list_path."
    assert feature_dir is not None, "Specify preprocess.feature_dir."

    if dump_format == "webdataset":
        template_path = os.path.join(feature_dir, "%d.tar")

        # TODO: support max_workers
        max_shard_size = config.preprocess.max_shard_size

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in tqdm(f):
                track_id = line.strip()
                track_id = int(track_id)
                process_webdataset(sink, track_id=track_id, fma_root=fma_root)
    else:
        raise NotImplementedError("Only dump_format=webdataset is supported.")


def process_webdataset(
    sink: wds.ShardWriter, track_id: str, fma_root: str, ext: str = "mp3"
) -> None:
    feature = {}

    name = track_id // 1000
    name = f"{name:03d}"
    mp3_path = os.path.join(fma_root, "audio", name, f"{track_id:06d}.{ext}")

    metadata = torchaudio.info(mp3_path, format=ext)

    with open(mp3_path, mode="rb") as f:
        audio = f.read()

    feature["__key__"] = str(track_id)
    feature[f"audio.{ext}"] = audio
    feature["filename.txt"] = str(track_id)
    feature["sample_rate.pth"] = torch.tensor(metadata.sample_rate, dtype=torch.long)

    sink.write(feature)


if __name__ == "__main__":
    main()
