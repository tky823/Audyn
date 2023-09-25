import os
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, List

import torch
import torchaudio
import torchaudio.functional as aF
import torchaudio.transforms as aT
import torchtext
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils.data.cmudict import BOS_SYMBOL, BREAK_SYMBOLS, EOS_SYMBOL
from audyn.utils.textgrid import load_textgrid


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    textgrid_dir = config.preprocess.textgrid_dir
    feature_dir = config.preprocess.feature_dir
    symbols_path = config.preprocess.symbols_path

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert textgrid_dir is not None, "Specify preprocess.textgrid_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."
    assert symbols_path is not None, "Specify preprocess.symbols_path."

    melspectrogram_transform = aT.MelSpectrogram(**config.data.melspectrogram)
    vocab = torch.load(symbols_path, map_location=lambda storage, loc: storage)

    os.makedirs(feature_dir, exist_ok=True)

    max_workers = config.preprocess.max_workers

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            with open(list_path) as f:
                for line in f:
                    filename = line.strip()
                    wav_path = os.path.join(wav_dir, f"{filename}.wav")
                    textgrid_path = os.path.join(textgrid_dir, f"{filename}.TextGrid")
                    feature_path = os.path.join(feature_dir, f"{filename}.pth")

                    future = executor.submit(
                        process,
                        filename=filename,
                        wav_path=wav_path,
                        textgrid_path=textgrid_path,
                        feature_path=feature_path,
                        melspectrogram_transform=melspectrogram_transform,
                        vocab=vocab,
                    )
                    futures.append(future)

            for future in tqdm(futures):
                future.result()
    else:
        with open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                wav_path = os.path.join(wav_dir, f"{filename}.wav")
                textgrid_path = os.path.join(textgrid_dir, f"{filename}.TextGrid")
                feature_path = os.path.join(feature_dir, f"{filename}.pth")

                process(
                    filename=filename,
                    wav_path=wav_path,
                    textgrid_path=textgrid_path,
                    feature_path=feature_path,
                    melspectrogram_transform=melspectrogram_transform,
                    vocab=vocab,
                )


def process(
    filename: str,
    wav_path: str,
    textgrid_path: str,
    feature_path: str,
    melspectrogram_transform: torchaudio.transforms.MelSpectrogram = None,
    vocab: torchtext.vocab.Vocab = None,
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

    alignment = load_textgrid(textgrid_path)
    alignment = postprocess_alignment(alignment)

    phones = []

    for data in alignment["phones"]:
        phones.append(data["text"])

    if phones[-1] in BREAK_SYMBOLS:
        phones[-1] = EOS_SYMBOL
    else:
        phones.append(EOS_SYMBOL)

    phones = vocab(phones)
    feature["phones"] = torch.tensor(phones, dtype=torch.long)
    feature["phones_length"] = torch.tensor(feature["phones"].size(-1), dtype=torch.long)
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
