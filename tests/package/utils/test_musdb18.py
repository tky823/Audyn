import glob
import os
import tempfile
import warnings
from datetime import timedelta
from typing import Dict, List

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchaudio
from dummy import allclose
from dummy.utils import select_random_port
from dummy.utils.ddp import set_ddp_environment
from torch.utils.data import DataLoader

from audyn.utils.data import Collator, Mixer
from audyn.utils.data.musdb18 import (
    sources,
    test_track_names,
    train_track_names,
    validation_track_names,
)
from audyn.utils.data.musdb18.dataset import (
    MUSDB18,
    DistributedRandomStemsMUSDB18Dataset,
    RandomStemsMUSDB18Dataset,
    StemsMUSDB18Dataset,
    Track,
)


def test_musdb18() -> None:
    sample_rate = 24000
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _ = _save_dummy_musdb18(root=temp_dir, sample_rate=sample_rate, num_frames=num_frames)

        # To ignore "Track is not found."
        warnings.simplefilter("ignore", UserWarning)
        dataset = MUSDB18(temp_dir, subset="train", ext="wav")
        warnings.resetwarnings()

        for track in dataset:
            track: Track
            track.frame_offset = num_frames // 4
            track.num_frames = num_frames // 2

            drums, _ = track.drums
            bass, _ = track.bass
            other, _ = track.other
            vocals, _ = track.vocals
            accompaniment, _ = track.accompaniment
            mixture, _ = track.mixture
            stems, _ = track.stems

            assert track.name in train_track_names
            allclose(drums + bass + other + vocals, mixture, atol=1e-4)
            allclose(accompaniment + vocals, mixture, atol=1e-4)
            allclose(stems[0], mixture, atol=1e-4)
            allclose(stems[1], drums)
            allclose(stems[2], bass)
            allclose(stems[3], other)
            allclose(stems[4], vocals)

            break


@pytest.mark.parametrize("replacement", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_musdb18_dataset(
    replacement: bool,
    num_workers: int,
) -> None:
    batch_size = 3
    sample_rate = 24000
    num_frames = 48000
    duration = 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        feature_dir = os.path.join(temp_dir, "features")
        list_dir = os.path.join(temp_dir, "list")
        train_feature_dir = os.path.join(feature_dir, "train")
        train_list_path = os.path.join(list_dir, "train.txt")

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)

        _ = _save_dummy_musdb18(root=feature_dir, sample_rate=sample_rate, num_frames=num_frames)

        with open(train_list_path, mode="w") as f:
            for track_name in sorted(glob.glob(os.path.join(train_feature_dir, "*"))):
                f.write(track_name + "\n")

        if not replacement:
            # StemsMUSDB18Dataset
            dataset = StemsMUSDB18Dataset(train_list_path, train_feature_dir, duration=duration)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

            filenames = []

            for feature in dataloader:
                assert set(feature.keys()) == {
                    "drums",
                    "bass",
                    "other",
                    "vocals",
                    "sample_rate",
                    "filename",
                }

                filenames_per_batch = set()

                for filename in feature["filename"]:
                    # If ``replacement=True``, filename may be included in filenames_per_batch
                    # with tiny probability.
                    assert filename not in filenames_per_batch

                    filenames_per_batch.add(filename)
                    filenames.append(filename)

            assert len(set(filenames)) == len(dataset.filenames)

        # RandomStemsMUSDB18Dataset
        dataset = RandomStemsMUSDB18Dataset(
            train_list_path, train_feature_dir, duration=duration, replacement=replacement
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        filenames = []

        for feature in dataloader:
            assert set(feature.keys()) == {
                "drums",
                "bass",
                "other",
                "vocals",
                "sample_rate",
                "filename",
            }

            filenames_per_batch = set()

            for filename in feature["filename"]:
                # If ``replacement=True``, filename may be included in filenames_per_batch
                # with tiny probability.
                assert filename not in filenames_per_batch

                filenames_per_batch.add(filename)
                filenames.append(filename)

        if not replacement:
            assert len(set(filenames)) == len(dataset.filenames)


@pytest.mark.parametrize("replacement", [True, False])
@pytest.mark.parametrize("num_workers", [1, 2])
@pytest.mark.parametrize("divisible_by_num_workers", [True, False])
def test_distributed_musdb18_dataset(
    replacement: bool,
    num_workers: int,
    divisible_by_num_workers: bool,
) -> None:
    # NOTE: Set num_workers=1 instead of num_workers=0 to prevent segment fault on Ubuntu.
    port = select_random_port()
    seed = 0
    world_size = 1

    torch.manual_seed(seed)

    epochs = 2
    sample_rate = 24000
    num_frames = 48000
    duration = 1
    input_keys = [
        "bass",
        "drums",
        "other",
        "vocals",
    ]
    output_key = "mixture"

    if divisible_by_num_workers:
        expected_samples_per_epoch = 8
        batch_size = 2
    else:
        expected_samples_per_epoch = 10
        batch_size = 3

    with tempfile.TemporaryDirectory() as temp_dir:
        list_dir = os.path.join(temp_dir, "list")
        feature_dir = os.path.join(temp_dir, "feature")
        train_feature_dir = os.path.join(feature_dir, "train")
        train_list_path = os.path.join(list_dir, "train.txt")

        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        track_names = _save_dummy_musdb18(
            root=feature_dir, sample_rate=sample_rate, num_frames=num_frames
        )
        train_track_names = track_names["train"]

        with open(train_list_path, mode="w") as f_list:
            for track_name in train_track_names:
                f_list.write(track_name + "\n")

        processes = []

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=run_distributed_musdb18_dataset_sampler,
                args=(rank, world_size, port),
                kwargs={
                    "replacement": replacement,
                    "samples_per_epoch": expected_samples_per_epoch,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "seed": seed,
                    "path": path,
                    "num_workers": num_workers,
                    "list_path": train_list_path,
                    "feature_dir": train_feature_dir,
                    "duration": duration,
                    "input_keys": input_keys,
                    "output_key": output_key,
                },
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        for epoch in range(epochs):
            rank = 0
            path = os.path.join(temp_dir, f"{rank}.pth")
            filenames_0 = torch.load(path)

            assert len(filenames_0[epoch]) == expected_samples_per_epoch

            for rank in range(1, world_size):
                path = os.path.join(temp_dir, f"{rank}.pth")
                filenames_rank = torch.load(path)

                # ensure disjointness among ranks
                assert filenames_0 != filenames_rank
                assert len(filenames_rank[epoch]) == expected_samples_per_epoch


def run_distributed_musdb18_dataset_sampler(
    rank: int,
    world_size: int,
    port: int,
    replacement: bool,
    samples_per_epoch: int,
    epochs: int,
    batch_size: int,
    seed: int = 0,
    path: str = None,
    num_workers: int = 1,
    list_path: str = None,
    feature_dir: str = None,
    duration: float = 1,
    input_keys: List[str] = None,
    output_key: str = None,
) -> None:
    set_ddp_environment(rank, world_size, port)

    dist.init_process_group(
        backend="gloo",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(minutes=5),
    )
    torch.manual_seed(seed)

    composer = Mixer(input_keys, output_key)
    collator = Collator(composer=composer)
    dataset = DistributedRandomStemsMUSDB18Dataset(
        list_path,
        feature_dir,
        duration,
        replacement=replacement,
        num_samples=samples_per_epoch,
        num_replicas=world_size,
        rank=rank,
        seed=seed,
    )
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

    dist.destroy_process_group()


def _save_dummy_musdb18(root: str, sample_rate: int, num_frames: int) -> Dict[str, List[str]]:
    g = torch.Generator()
    g.manual_seed(0)

    _train_track_names = train_track_names[:10]
    _validation_track_names = validation_track_names[:10]
    _test_track_names = test_track_names[:10]

    track_names = {
        "train": _train_track_names,
        "validation": _validation_track_names,
        "test": _test_track_names,
    }

    num_channels = 2
    subset_name = "train"

    for track_name in _train_track_names + _validation_track_names:
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        mixture = 0

        for source in sources:
            path = os.path.join(track_dir, f"{source}.wav")
            waveform = torch.randn((num_channels, num_frames), generator=g)
            max_amplitude = torch.max(torch.abs(waveform))
            waveform = 0.1 * (waveform / max_amplitude)
            mixture = mixture + waveform
            torchaudio.save(path, waveform, sample_rate)

        path = os.path.join(track_dir, "mixture.wav")
        torchaudio.save(path, mixture, sample_rate)

    subset_name = "test"

    for track_name in _test_track_names:
        track_dir = os.path.join(root, subset_name, track_name)

        os.makedirs(track_dir, exist_ok=True)

        mixture = 0

        for source in sources:
            path = os.path.join(track_dir, f"{source}.wav")
            waveform = torch.randn((num_channels, num_frames), generator=g)
            max_amplitude = torch.max(torch.abs(waveform))
            waveform = 0.1 * (waveform / max_amplitude)
            mixture = mixture + waveform
            torchaudio.save(path, waveform, sample_rate)

        path = os.path.join(track_dir, "mixture.wav")
        torchaudio.save(path, mixture, sample_rate)

    return track_names
