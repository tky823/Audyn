import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT
import torchtext
from omegaconf import DictConfig
from tqdm import tqdm

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    feature_dir = config.preprocess.feature_dir
    category_path = config.preprocess.category_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert category_path is not None, "Specify preprocess.category_path."

    melspectrogram_transform = aT.MelSpectrogram(**config.data.melspectrogram)
    category_to_id = torch.load(
        category_path,
        map_location=lambda storage, loc: storage,
        weights_only=True,
    )

    os.makedirs(feature_dir, exist_ok=True)

    max_workers = config.preprocess.max_workers

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            with open(list_path) as f:
                for line in f:
                    filename = line.strip()
                    wav_path = os.path.join(wav_dir, f"{filename}.wav")
                    *_, category, _ = filename.split("/")
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")

                    future = executor.submit(
                        process,
                        filename=filename,
                        wav_path=wav_path,
                        feature_path=feature_path,
                        category=category,
                        melspectrogram_transform=melspectrogram_transform,
                        category_to_id=category_to_id,
                    )
                    futures.append(future)

            for future in tqdm(futures):
                future.result()
    else:
        with open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                wav_path = os.path.join(wav_dir, f"{filename}.wav")
                *_, category, _ = filename.split("/")
                feature_path = os.path.join(feature_dir, f"{filename}.pth")

                process(
                    filename=filename,
                    wav_path=wav_path,
                    feature_path=feature_path,
                    category=category,
                    melspectrogram_transform=melspectrogram_transform,
                    category_to_id=category_to_id,
                )


def process(
    filename: str,
    wav_path: str,
    feature_path: str,
    category: str,
    melspectrogram_transform: aT.MelSpectrogram = None,
    category_to_id: torchtext.vocab.Vocab = None,
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
    feature["category"] = category_to_id[category]

    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


if __name__ == "__main__":
    main()
