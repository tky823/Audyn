import os
import sys
import tempfile
from datetime import timedelta
from typing import Any, Dict, Optional

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
import webdataset as wds
from dummy.utils import select_random_port
from dummy.utils.ddp import set_ddp_environment
from torch.utils.data import DataLoader

from audyn.transforms.ast import AudioSpectrogramTransformerMelSpectrogram
from audyn.utils.data.audioset.composer import (
    ASTAudioSetMultiLabelComposer,
    AudioSetMultiLabelComposer,
)
from audyn.utils.data.audioset.dataset import (
    DistributedPaSSTAudioSetWebDataset,
    DistributedWeightedAudioSetWebDataset,
    PaSSTAudioSetWebDataset,
    WeightedAudioSetWebDataset,
)
from audyn.utils.data.collator import Collator

IS_WINDOWS = sys.platform == "win32"


@pytest.mark.parametrize("dataset_type", [None, "PaSST"])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("divisible_by_num_workers", [True, False])
@pytest.mark.parametrize("composer_pattern", [1, 2])
def test_weighted_audioset_webdataset(
    audioset_samples: Dict[str, Dict[str, Any]],
    dataset_type: Optional[str],
    num_workers: int,
    divisible_by_num_workers: bool,
    composer_pattern: int,
) -> None:
    torch.manual_seed(0)

    max_shard_count = 4
    audio_key, sample_rate_key, filename_key = "audio", "sample_rate", "filename"
    tags_key, multilabel_key = "tags", "tags_index"

    n_mels, n_frames = 40, 100

    if divisible_by_num_workers:
        expected_samples_per_epoch = 4
        batch_size = 2
    else:
        expected_samples_per_epoch = 5
        batch_size = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        list_path = os.path.join(list_dir, "train.txt")
        tar_path = os.path.join(feature_dir, "%d.tar")

        if IS_WINDOWS:
            # https://stackoverflow.com/questions/68299665/valueerror-no-gopen-handler-defined
            feature_dir = "file:" + feature_dir
            tar_path = "file:" + tar_path

        with (
            wds.ShardWriter(tar_path, maxcount=max_shard_count) as sink,
            open(list_path, mode="w") as f_list,
        ):
            for ytid in sorted(audioset_samples.keys()):
                sample = audioset_samples[ytid]
                sample_rate = sample["sample_rate"]
                tags = sample["tags"]
                waveform = torch.randn((2, 10 * sample_rate))
                amplitude = torch.abs(waveform)
                waveform = waveform / torch.max(amplitude)
                waveform = 0.9 * waveform
                path = os.path.join(audio_dir, f"{ytid}.wav")
                torchaudio.save(path, waveform, sample_rate)

                with open(path, mode="rb") as f_audio:
                    audio = f_audio.read()

                feature = {
                    "__key__": ytid,
                    f"{audio_key}.wav": audio,
                    f"{tags_key}.json": tags,
                    f"{filename_key}.txt": ytid,
                    f"{sample_rate_key}.pth": torch.tensor(sample_rate, dtype=torch.long),
                }

                sink.write(feature)
                f_list.write(ytid + "\n")

        assert len(os.listdir(feature_dir)) == (len(audioset_samples) - 1) // max_shard_count + 1

        if dataset_type is None:
            dataset_cls = WeightedAudioSetWebDataset
            composer = AudioSetMultiLabelComposer(tags_key, multilabel_key)
        elif dataset_type.lower() == "passt":
            dataset_cls = PaSSTAudioSetWebDataset
            melspectrogram_transform = AudioSpectrogramTransformerMelSpectrogram(
                sample_rate,
                n_mels=n_mels,
                n_frames=n_frames,
            )
            composer = ASTAudioSetMultiLabelComposer(
                melspectrogram_transform,
                audio_key=audio_key,
                sample_rate_key=sample_rate_key,
                tags_key=tags_key,
                filename_key=filename_key,
                multilabel_key=multilabel_key,
            )
        else:
            raise ValueError(f"Invalid dataset type {dataset_type} is given.")

        if composer_pattern == 1:
            # pattern 1: set composer to dataset
            collator = Collator()
            dataset = dataset_cls.instantiate_dataset(
                list_path,
                feature_dir,
                length=expected_samples_per_epoch,
                composer=composer,
            )
        elif composer_pattern == 2:
            # pattern 2: set composer to collator
            collator = Collator(composer=composer)
            dataset = dataset_cls(
                list_path,
                feature_dir,
                length=expected_samples_per_epoch,
            )
        else:
            raise ValueError(f"Invalid composer_pattern {composer_pattern} is given.")

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
        )

        samples_per_epoch = 0

        for sample in dataloader:
            samples_per_epoch += len(sample["filename"])

        assert samples_per_epoch == expected_samples_per_epoch

        dataset.close_all()


@pytest.mark.parametrize("dataset_type", [None, "PaSST"])
@pytest.mark.parametrize("num_workers", [0, 2])
@pytest.mark.parametrize("divisible_by_num_workers", [True, False])
@pytest.mark.parametrize("composer_pattern", [1, 2])
def test_distributed_weighted_audioset_webdataset_sampler(
    audioset_samples: Dict[str, Dict[str, Any]],
    dataset_type: Optional[str],
    num_workers: int,
    divisible_by_num_workers: bool,
    composer_pattern: int,
) -> None:
    port = select_random_port()
    seed = 0
    world_size = 2

    torch.manual_seed(seed)

    epochs = 2
    max_shard_count = 4
    tags_key, multilabel_key = "tags", "tags_index"

    if divisible_by_num_workers:
        expected_samples_per_epoch = 8
        batch_size = 2
    else:
        expected_samples_per_epoch = 10
        batch_size = 3

    with tempfile.TemporaryDirectory(dir=".") as temp_dir:
        audio_dir = os.path.join(temp_dir, "audio")
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")

        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        list_path = os.path.join(list_dir, "train.txt")
        tar_path = os.path.join(feature_dir, "%d.tar")

        if IS_WINDOWS:
            # https://stackoverflow.com/questions/68299665/valueerror-no-gopen-handler-defined
            feature_dir = "file:" + feature_dir
            tar_path = "file:" + tar_path

        with (
            wds.ShardWriter(tar_path, maxcount=max_shard_count) as sink,
            open(list_path, mode="w") as f_list,
        ):
            for ytid in sorted(audioset_samples.keys()):
                sample = audioset_samples[ytid]
                sample_rate = sample["sample_rate"]
                tags = sample["tags"]
                waveform = torch.randn((2, 10 * sample_rate))
                amplitude = torch.abs(waveform)
                waveform = waveform / torch.max(amplitude)
                waveform = 0.9 * waveform
                path = os.path.join(audio_dir, f"{ytid}.wav")
                torchaudio.save(path, waveform, sample_rate)

                with open(path, mode="rb") as f_audio:
                    audio = f_audio.read()

                feature = {
                    "__key__": ytid,
                    "audio.wav": audio,
                    f"{tags_key}.json": tags,
                    "filename.txt": ytid,
                    "sample_rate.pth": torch.tensor(sample_rate, dtype=torch.long),
                }

                sink.write(feature)
                f_list.write(ytid + "\n")

        assert len(os.listdir(feature_dir)) == (len(audioset_samples) - 1) // max_shard_count + 1

        processes = []

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=run_distributed_weighted_audioset_webdataset_sampler,
                args=(rank, world_size, port),
                kwargs={
                    "samples_per_epoch": expected_samples_per_epoch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                    "path": path,
                    "dataset_type": dataset_type,
                    "num_workers": num_workers,
                    "list_path": list_path,
                    "feature_dir": feature_dir,
                    "tags_key": tags_key,
                    "multilabel_key": multilabel_key,
                    "composer_pattern": composer_pattern,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for epoch in range(epochs):
            samples_per_epoch = 0
            rank = 0
            path = os.path.join(temp_dir, f"{rank}.pth")
            filenames_0 = torch.load(path)
            samples_per_epoch += len(filenames_0[epoch])

            for rank in range(1, world_size):
                path = os.path.join(temp_dir, f"{rank}.pth")
                filenames_rank = torch.load(path)
                samples_per_epoch += len(filenames_rank[epoch])

                # ensure disjointness among ranks
                assert filenames_0 != filenames_rank

            assert samples_per_epoch == expected_samples_per_epoch


@pytest.fixture
def audioset_samples() -> Dict[str, Dict[str, Any]]:
    sample_rate = 44100

    samples = {
        "example0": {
            "tags": ["/m/09x0r", "/m/05zppz"],
            "sample_rate": sample_rate,
        },
        "example1": {
            "tags": ["/m/02zsn"],
            "sample_rate": sample_rate,
        },
        "example2": {
            "tags": ["/m/05zppz", "/m/0ytgt", "/m/01h8n0", "/m/02qldy"],
            "sample_rate": sample_rate,
        },
        "example3": {
            "tags": ["/m/0261r1"],
            "sample_rate": sample_rate,
        },
        "example4": {
            "tags": ["/m/0261r1", "/m/09x0r", "/m/0brhx"],
            "sample_rate": sample_rate,
        },
        "example5": {
            "tags": ["/m/0261r1", "/m/07p6fty"],
            "sample_rate": sample_rate,
        },
        "example6": {
            "tags": ["/m/09x0r", "/m/07q4ntr"],
            "sample_rate": sample_rate,
        },
        "example7": {
            "tags": ["/m/09x0r", "/m/07rwj3x", "/m/07sr1lc"],
            "sample_rate": sample_rate,
        },
        "example8": {
            "tags": ["/m/04gy_2"],
            "sample_rate": sample_rate,
        },
        "example9": {
            "tags": ["/t/dd00135", "/m/03qc9zr", "/m/02rtxlg", "/m/01j3sz"],
            "sample_rate": sample_rate,
        },
        "example10": {
            "tags": ["/m/0261r1", "/m/05zppz", "/t/dd00001"],
            "sample_rate": sample_rate,
        },
    }

    return samples


def run_distributed_weighted_audioset_webdataset_sampler(
    rank: int,
    world_size: int,
    port: int,
    samples_per_epoch: int,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    path: str = None,
    dataset_type: Optional[str] = None,
    num_workers: int = 1,
    list_path: str = None,
    feature_dir: str = None,
    tags_key: str = None,
    multilabel_key: str = None,
    composer_pattern: int = 1,
) -> None:
    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    torch.manual_seed(seed)

    if dataset_type is None:
        dataset_cls = DistributedWeightedAudioSetWebDataset
    elif dataset_type.lower() == "passt":
        dataset_cls = DistributedPaSSTAudioSetWebDataset
    else:
        raise ValueError(f"Invalid dataset type {dataset_type} is given.")

    composer = AudioSetMultiLabelComposer(tags_key, multilabel_key)

    if composer_pattern == 1:
        # pattern 1: set composer to dataset
        collator = Collator()
        dataset = dataset_cls.instantiate_dataset(
            list_path,
            feature_dir,
            length=samples_per_epoch,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
            composer=composer,
        )
    elif composer_pattern == 2:
        # pattern 2: set composer to collator
        collator = Collator(composer=composer)
        dataset = dataset_cls(
            list_path,
            feature_dir,
            length=samples_per_epoch,
            num_replicas=world_size,
            rank=rank,
            seed=seed,
        )
    else:
        raise ValueError(f"Invalid composer_pattern {composer_pattern} is given.")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collator,
    )

    filenames = []

    for _ in range(epochs):
        filenames_per_epoch = []

        for sample in dataloader:
            filenames_per_epoch.extend(sample["filename"])

        filenames.append(filenames_per_epoch)

    torch.save(filenames, path)

    dataset.close_all()

    dist.destroy_process_group()
