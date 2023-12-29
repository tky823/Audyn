import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import torchaudio
import torchaudio.functional as aF
from omegaconf import DictConfig
from tqdm import tqdm

import audyn


@audyn.main()
def main(config: DictConfig) -> None:
    list_path = config.preprocess.list_path
    wav_dir = config.preprocess.wav_dir

    assert list_path is not None, "Specify preprocess.list_path."
    assert wav_dir is not None, "Specify preprocess.wav_dir."

    sample_rate = config.data.audio.sample_rate
    slice_length = config.data.audio.slice_length

    max_workers = config.preprocess.max_workers
    valid_filenames = []

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []

            with open(list_path) as f:
                for line in f:
                    filename = line.strip()
                    wav_path = os.path.join(wav_dir, f"{filename}.wav")

                    future = executor.submit(
                        process,
                        filename=filename,
                        wav_path=wav_path,
                        sample_rate=sample_rate,
                        slice_length=slice_length,
                    )
                    futures.append(future)

            for future in tqdm(futures):
                filename = future.result()

                if filename is not None:
                    valid_filenames.append(filename)
    else:
        with open(list_path) as f:
            for line in tqdm(f):
                filename = line.strip()
                wav_path = os.path.join(wav_dir, f"{filename}.wav")

                filename = process(
                    filename=filename,
                    wav_path=wav_path,
                    sample_rate=sample_rate,
                    slice_length=slice_length,
                )

                if filename is not None:
                    valid_filenames.append(filename)

    with open(list_path, mode="w") as f:
        for filename in valid_filenames:
            f.write(filename + "\n")


def process(
    filename: str,
    wav_path: str,
    sample_rate: Optional[int] = None,
    slice_length: Optional[int] = None,
) -> Optional[str]:
    waveform, _sample_rate = torchaudio.load(wav_path)

    if sample_rate is None and _sample_rate != sample_rate:
        # TODO: torchaudio.transforms.Resample
        waveform = aF.resample(waveform, _sample_rate, sample_rate)

    if waveform.size(-1) < slice_length:
        return None

    return filename


if __name__ == "__main__":
    main()
