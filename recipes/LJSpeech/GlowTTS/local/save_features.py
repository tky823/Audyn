import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import hydra
import torch
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils.data.cmudict import BOS_SYMBOL, BREAK_SYMBOLS, EOS_SYMBOL
from audyn.utils.text import TextPreprocessor, load_text


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

    melspectrogram_transform = aT.MelSpectrogram(**config.data.melspectrogram)
    text_preprocessor = hydra.utils.instantiate(config.data.text.preprocessor)

    os.makedirs(feature_dir, exist_ok=True)

    max_workers = config.preprocess.max_workers

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            with open(list_path) as f:
                for line in f:
                    filename = line.strip()
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
                filename = line.strip()
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
        # TODO: aT.Resample
        waveform = aF.resample(waveform, sample_rate, melspectrogram_transform.sample_rate)
        sample_rate = melspectrogram_transform.sample_rate

    waveform = waveform.mean(dim=0)
    melspectrogram = melspectrogram_transform(waveform)

    feature["waveform"] = waveform
    feature["melspectrogram"] = melspectrogram
    feature["waveform_length"] = torch.tensor(waveform.size(-1), dtype=torch.long)
    feature["melspectrogram_length"] = torch.tensor(melspectrogram.size(-1), dtype=torch.long)

    # Text normalization should be applied in advance.
    normalized_sentence = load_text(text_path)
    phones = text_preprocessor.index_normalized_text(normalized_sentence, return_type="tensor")

    feature["phones"] = phones
    feature["phones_length"] = torch.tensor(phones.size(-1), dtype=torch.long)
    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def postprocess_alignment(
    alignment: Dict[str, List[Dict[str, str]]]
) -> Dict[str, List[Dict[str, str]]]:
    # replace "" with special symbol
    for name in alignment:
        for idx, data in enumerate(alignment[name]):
            if data["text"] == "":
                if idx == 0:
                    data["text"] = "sil"  # silent at BOS
                elif idx == len(alignment[name]) - 1:
                    data["text"] = "sil"  # silent at EOS
                else:
                    data["text"] = "spn"  # spoken

    return alignment


def postprocess_phones(text: List[str]) -> List[str]:
    # insert BOS and EOS
    assert text[0] not in BREAK_SYMBOLS
    assert text[-1] not in BREAK_SYMBOLS

    if text[0] != BOS_SYMBOL:
        text.insert(0, BOS_SYMBOL)

    if text[-1] != EOS_SYMBOL:
        text.append(EOS_SYMBOL)

    return text


if __name__ == "__main__":
    main()
