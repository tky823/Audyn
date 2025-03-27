import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torch
import torchaudio
import torchaudio.functional as aF
import webdataset as wds
from omegaconf import DictConfig
from tqdm import tqdm

import audyn
from audyn.utils import instantiate, setup_config
from audyn.utils.text import TextPreprocessor, load_text


@audyn.main()
def main(config: DictConfig) -> None:
    setup_config(config)

    dump_format = config.preprocess.dump_format
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir
    text_dir = config.preprocess.text_dir
    feature_dir = config.preprocess.feature_dir

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."
    assert text_dir is not None, "Specify preprocess.text_dir."
    assert feature_dir is not None, "Specify preprocess.feature_dir."

    sample_rate = config.data.audio.sample_rate
    bos_token = config.data.text.bos_token
    eos_token = config.data.text.eos_token

    text_preprocessor = instantiate(config.data.text.preprocessor)

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
                        text_path = os.path.join(text_dir, f"{filename}.txt")
                        feature_path = os.path.join(feature_dir, f"{filename}.pth")

                        future = executor.submit(
                            process_torch,
                            filename=filename,
                            wav_path=wav_path,
                            text_path=text_path,
                            feature_path=feature_path,
                            text_preprocessor=text_preprocessor,
                            sample_rate=sample_rate,
                            bos_token=bos_token,
                            eos_token=eos_token,
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

                    process_torch(
                        filename=filename,
                        wav_path=wav_path,
                        text_path=text_path,
                        feature_path=feature_path,
                        text_preprocessor=text_preprocessor,
                        sample_rate=sample_rate,
                        bos_token=bos_token,
                        eos_token=eos_token,
                    )
    else:
        template_path = os.path.join(feature_dir, "%d.tar")

        max_shard_size = config.preprocess.max_shard_size

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink, open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                text_path = os.path.join(text_dir, f"{filename}.txt")
                wav_path = os.path.join(wav_dir, f"{filename}.wav")

                process_webdataset(
                    sink,
                    filename=filename,
                    wav_path=wav_path,
                    text_path=text_path,
                    text_preprocessor=text_preprocessor,
                    sample_rate=sample_rate,
                    bos_token=bos_token,
                    eos_token=eos_token,
                )


def process_torch(
    filename: str,
    wav_path: str,
    text_path: str,
    feature_path: str,
    text_preprocessor: TextPreprocessor = None,
    sample_rate: Optional[int] = None,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
) -> None:
    feature = {}

    # audio
    waveform, _sample_rate = torchaudio.load(wav_path)

    if sample_rate is not None and _sample_rate != sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, _sample_rate, sample_rate)

    waveform = waveform.mean(dim=0)

    feature["waveform"] = waveform
    feature["waveform_length"] = torch.tensor(waveform.size(-1), dtype=torch.long)

    # text
    normalized_text = load_text(text_path)
    phones = text_preprocessor.phonemize_normalized_text(normalized_text)

    if bos_token is not None and phones[0] != bos_token:
        phones.insert(0, bos_token)

    if eos_token is not None and phones[-1] != eos_token:
        phones.append(eos_token)

    phones = text_preprocessor.index_phonemes(phones, return_type="tensor")
    feature["phones"] = phones
    feature["phones_length"] = torch.tensor(phones.size(-1), dtype=torch.long)

    feature["filename"] = filename

    feature_dir = os.path.dirname(feature_path)
    os.makedirs(feature_dir, exist_ok=True)
    torch.save(feature, feature_path)


def process_webdataset(
    sink: wds.ShardWriter,
    filename: str,
    wav_path: str,
    text_path: str,
    text_preprocessor: TextPreprocessor = None,
    sample_rate: Optional[int] = None,
    bos_token: Optional[str] = None,
    eos_token: Optional[str] = None,
) -> None:
    feature = {}

    feature["__key__"] = filename

    # audio
    waveform, _sample_rate = torchaudio.load(wav_path)

    if sample_rate is None and _sample_rate != sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, _sample_rate, sample_rate)

    waveform = waveform.mean(dim=0)

    feature["waveform.pth"] = waveform
    feature["waveform_length.pth"] = torch.tensor(waveform.size(-1), dtype=torch.long)

    # text
    normalized_text = load_text(text_path)
    phones = text_preprocessor.phonemize_normalized_text(normalized_text)

    if bos_token is not None and phones[0] != bos_token:
        phones.insert(0, bos_token)

    if eos_token is not None and phones[-1] != eos_token:
        phones.append(eos_token)

    phones = text_preprocessor.index_phonemes(phones, return_type="tensor")
    feature["phones.pth"] = phones
    feature["phones_length.pth"] = torch.tensor(phones.size(-1), dtype=torch.long)

    feature["filename.txt"] = filename

    sink.write(feature)


if __name__ == "__main__":
    main()
