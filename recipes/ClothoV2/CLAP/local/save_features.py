import os
from concurrent.futures import ProcessPoolExecutor

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import instantiate
from audyn.utils.text import TextPreprocessor


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    text_dir = config.preprocess.text_dir
    feature_dir = config.preprocess.feature_dir

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert text_dir is not None, "Specify preprocess.text_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."

    melspectrogram_transform = instantiate(config.data.melspectrogram)
    text_preprocessor = instantiate(config.data.text.preprocessor)

    os.makedirs(feature_dir, exist_ok=True)

    max_workers = config.preprocess.max_workers

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            with open(list_path) as f:
                for line in f:
                    filename = line.strip("\n")
                    wav_path = os.path.join(wav_dir, f"{filename}.wav")
                    text_path = os.path.join(text_dir, f"{filename}.txt")
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")

                    future = executor.submit(
                        process,
                        filename=filename,
                        wav_path=wav_path,
                        text_path=text_path,
                        feature_path=feature_path,
                        melspectrogram_transform=melspectrogram_transform,
                        text_preprocessor=text_preprocessor,
                    )
                    futures.append(future)

            for future in tqdm(futures):
                future.result()
    else:
        with open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip("\n")
                wav_path = os.path.join(wav_dir, f"{filename}.wav")
                text_path = os.path.join(text_dir, f"{filename}.txt")
                feature_path = os.path.join(feature_dir, f"{filename}.pth")

                process(
                    filename=filename,
                    wav_path=wav_path,
                    text_path=text_path,
                    feature_path=feature_path,
                    melspectrogram_transform=melspectrogram_transform,
                    text_preprocessor=text_preprocessor,
                )


def process(
    filename: str,
    wav_path: str,
    text_path: str,
    feature_path: str,
    melspectrogram_transform: aT.MelSpectrogram = None,
    text_preprocessor: TextPreprocessor = None,
) -> None:
    feature = {}

    waveform, sample_rate = torchaudio.load(wav_path)

    if sample_rate != melspectrogram_transform.sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, sample_rate, melspectrogram_transform.sample_rate)
        sample_rate = melspectrogram_transform.sample_rate

    waveform = waveform.mean(dim=0)
    melspectrogram = melspectrogram_transform(waveform)

    feature["waveform"] = waveform
    feature["melspectrogram"] = melspectrogram
    feature["waveform_length"] = torch.tensor(waveform.size(-1), dtype=torch.long)
    feature["melspectrogram_length"] = torch.tensor(melspectrogram.size(-1), dtype=torch.long)

    batched_tokens = []

    with open(text_path) as f:
        for line in f:
            caption = line.strip()
            tokens = text_preprocessor(caption, return_type="tensor")
            batched_tokens.append(tokens)

    feature["tokens"] = nn.utils.rnn.pack_sequence(batched_tokens, enforce_sorted=False)
    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


if __name__ == "__main__":
    main()
