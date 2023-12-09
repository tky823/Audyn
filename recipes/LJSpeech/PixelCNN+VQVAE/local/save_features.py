import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    feature_dir = config.preprocess.feature_dir

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."

    melspectrogram_transform = aT.MelSpectrogram(**config.data.melspectrogram)

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
                            melspectrogram_transform=melspectrogram_transform,
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
                        melspectrogram_transform=melspectrogram_transform,
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
                    melspectrogram_transform=melspectrogram_transform,
                )


def process_torch(
    filename: str,
    wav_path: str,
    feature_path: str,
    melspectrogram_transform: aT.MelSpectrogram = None,
) -> None:
    feature = {}

    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != melspectrogram_transform.sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, sample_rate, melspectrogram_transform.sample_rate)
        sample_rate = melspectrogram_transform.sample_rate

    waveform = waveform.mean(dim=0)
    melspectrogram = melspectrogram_transform(waveform)

    feature["melspectrogram"] = melspectrogram
    feature["melspectrogram_length"] = torch.tensor(melspectrogram.size(-1), dtype=torch.long)

    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    sink: wds.ShardWriter,
    filename: str,
    wav_path: str,
    melspectrogram_transform: aT.MelSpectrogram = None,
) -> None:
    feature = {}

    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != melspectrogram_transform.sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, sample_rate, melspectrogram_transform.sample_rate)
        sample_rate = melspectrogram_transform.sample_rate

    waveform = waveform.mean(dim=0)
    melspectrogram = melspectrogram_transform(waveform)

    feature["__key__"] = filename

    feature["melspectrogram.pth"] = melspectrogram
    feature["melspectrogram_length.pth"] = torch.tensor(melspectrogram.size(-1), dtype=torch.long)

    feature["filename.txt"] = filename

    sink.write(feature)


if __name__ == "__main__":
    main()
