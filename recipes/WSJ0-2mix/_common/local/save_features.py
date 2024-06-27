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

    num_sources = 2
    filenames = []

    with open(list_path) as f:
        for line in f:
            filename = line.strip()
            filenames.append(filename)

    if dump_format == "torch":
        max_workers = config.preprocess.max_workers

        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []

                for filename in filenames:
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")
                    future = executor.submit(
                        process_torch,
                        filename,
                        wav_dir=wav_dir,
                        feature_path=feature_path,
                        num_sources=num_sources,
                    )
                    futures.append(future)

                for future in tqdm(futures):
                    future.result()
        else:
            for filename in tqdm(filenames):
                feature_path = os.path.join(feature_dir, f"{filename}.pth")
                process_torch(
                    filename, wav_dir=wav_dir, feature_path=feature_path, num_sources=num_sources
                )
    elif dump_format == "webdataset":
        template_path = os.path.join(feature_dir, "%d.tar")
        max_shard_size = config.preprocess.max_shard_size

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()

                process_webdataset(
                    sink,
                    filename=filename,
                    wav_dir=wav_dir,
                )
    else:
        raise ValueError(f"{dump_format} is not supported.")


def process_torch(filename: str, wav_dir: str, feature_path: str, num_sources: int = 2) -> None:
    feature = {}

    feature["filename"] = filename

    path = os.path.join(wav_dir, "mix", f"{filename}.wav")
    waveform, sample_rate = torchaudio.load(path)
    feature["sample_rate"] = torch.tensor(sample_rate)
    feature["mixture"] = waveform.mean(dim=0)

    sources = []

    for idx in range(1, num_sources + 1):
        path = os.path.join(wav_dir, f"s{idx}", f"{filename}.wav")
        waveform, sample_rate = torchaudio.load(path)

        assert feature["sample_rate"].item() == sample_rate

        waveform = waveform.mean(dim=0)
        sources.append(waveform)

    feature["sources"] = torch.stack(sources, dim=0)

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    sink: wds.ShardWriter, filename: str, wav_dir: str, num_sources: int = 2
) -> None:
    feature = {}

    feature["__key__"] = filename
    feature["filename.txt"] = filename

    path = os.path.join(wav_dir, "mix", f"{filename}.wav")
    waveform, sample_rate = torchaudio.load(path)
    feature["sample_rate.pth"] = torch.tensor(sample_rate, dtype=torch.long)
    feature["mixture.pth"] = waveform.mean(dim=0)

    sources = []

    for idx in range(1, num_sources + 1):
        path = os.path.join(wav_dir, f"s{idx}", f"{filename}.wav")
        waveform, sample_rate = torchaudio.load(path)

        assert feature["sample_rate.pth"].item() == sample_rate

        waveform = waveform.mean(dim=0)
        sources.append(waveform)

    feature["sources.pth"] = torch.stack(sources, dim=0)

    sink.write(feature)


if __name__ == "__main__":
    main()
