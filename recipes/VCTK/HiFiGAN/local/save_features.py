import os
from concurrent.futures import ProcessPoolExecutor

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
    wav_dir = config.preprocess.wav_dir
    feature_dir = config.preprocess.feature_dir
    max_workers = config.preprocess.max_workers
    max_shard_size = config.preprocess.max_shard_size

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."

    os.makedirs(feature_dir, exist_ok=True)

    if dump_format == "torch":
        max_workers = config.preprocess.max_workers

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                with open(list_path) as f:
                    for line in f:
                        filename = line.strip()
                        wav_path = os.path.join(wav_dir, f"{filename}.wav")
                        feature_path = os.path.join(feature_dir, f"{filename}.pth")

                        future = executor.submit(
                            process_torch,
                            filename=filename,
                            wav_path=wav_path,
                            feature_path=feature_path,
                        )
                        futures.append(future)

                for future in tqdm(futures):
                    future.result()
        else:
            with open(list_path) as f:
                for line in tqdm(f):
                    filename = line.strip()
                    wav_path = os.path.join(wav_dir, f"{filename}.wav")
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")

                    process_torch(
                        filename=filename,
                        wav_path=wav_path,
                        feature_path=feature_path,
                    )
    else:
        template_path = os.path.join(feature_dir, "%d.tar")

        max_shard_size = config.preprocess.max_shard_size

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                wav_path = os.path.join(wav_dir, f"{filename}.wav")

                process_webdataset(
                    sink,
                    filename=filename,
                    wav_path=wav_path,
                )


def process_torch(
    filename: str,
    wav_path: str,
    feature_path: str,
) -> None:
    feature = {}

    wav_dir = os.path.dirname(wav_path)
    speaker = os.path.basename(wav_dir)
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)

    feature["waveform"] = waveform
    feature["sample_rate"] = torch.tensor(sample_rate, dtype=torch.long)
    feature["speaker"] = speaker
    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    sink: wds.ShardWriter,
    filename: str,
    wav_path: str,
) -> None:
    feature = {}

    wav_dir = os.path.dirname(wav_path)
    speaker = os.path.basename(wav_dir)
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.mean(dim=0)

    feature["__key__"] = filename
    feature["waveform.pth"] = waveform
    feature["sample_rate.pth"] = torch.tensor(sample_rate, dtype=torch.long)
    feature["speaker.txt"] = speaker
    feature["filename.txt"] = filename

    sink.write(feature)


if __name__ == "__main__":
    main()
