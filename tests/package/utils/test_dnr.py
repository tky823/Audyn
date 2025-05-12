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
from audyn_test import allclose
from audyn_test.utils import select_random_port
from audyn_test.utils.ddp import set_ddp_environment
from torch.utils.data import DataLoader

from audyn.utils.data import Collator, Mixer
from audyn.utils.data.dnr import (
    sources,
    v2_test_track_names,
    v2_train_track_names,
    v2_validation_track_names,
)
from audyn.utils.data.dnr.dataset import (
    DNR,
    DistributedRandomStemsDNRDataset,
    RandomStemsDNRDataset,
    StemsDNRDataset,
    Track,
)


def test_dnr() -> None:
    sample_rate = 24000
    num_frames = 48000

    with tempfile.TemporaryDirectory() as temp_dir:
        _ = _save_dummy_dnr(root=temp_dir, sample_rate=sample_rate, num_frames=num_frames)

        # To ignore "Track is not found."
        warnings.simplefilter("ignore", UserWarning)
        dataset = DNR(temp_dir, subset="train")
        warnings.resetwarnings()

        for track in dataset:
            track: Track
            track.frame_offset = num_frames // 4
            track.num_frames = num_frames // 2

            speech, _ = track.speech
            music, _ = track.music
            effect, _ = track.effect
            mixture, _ = track.mixture
            stems, _ = track.stems

            assert track.name in v2_train_track_names
            allclose(speech + music + effect, mixture, atol=1e-4)
            allclose(stems[0], mixture, atol=1e-4)
            allclose(stems[1], speech)
            allclose(stems[2], music)
            allclose(stems[3], effect)

            break


@pytest.mark.parametrize("replacement", [True, False])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_dnr_dataset(
    replacement: bool,
    num_workers: int,
) -> None:
    if not replacement:
        pytest.skip("replacement=False is not supported.")

    batch_size = 3
    sample_rate = 24000
    num_frames = 48000
    duration = 1.0

    with tempfile.TemporaryDirectory() as temp_dir:
        feature_dir = os.path.join(temp_dir, "features")
        list_dir = os.path.join(temp_dir, "list")
        train_feature_dir = os.path.join(feature_dir, "tr")
        train_list_path = os.path.join(list_dir, "train.txt")

        os.makedirs(feature_dir, exist_ok=True)
        os.makedirs(list_dir, exist_ok=True)

        _ = _save_dummy_dnr(root=feature_dir, sample_rate=sample_rate, num_frames=num_frames)

        with open(train_list_path, mode="w") as f:
            for track_name in sorted(glob.glob(os.path.join(train_feature_dir, "*"))):
                f.write(track_name + "\n")

        if not replacement:
            # StemsDNRDataset
            dataset = StemsDNRDataset(train_list_path, train_feature_dir, duration=duration)
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

            filenames = []

            for feature in dataloader:
                assert set(feature.keys()) == {
                    "speech",
                    "music",
                    "effect",
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

        # RandomStemsDNRDataset
        dataset = RandomStemsDNRDataset(
            train_list_path, train_feature_dir, duration=duration, replacement=replacement
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

        filenames = []

        for feature in dataloader:
            assert set(feature.keys()) == {
                "speech",
                "music",
                "effect",
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
def test_distributed_dnr_dataset(
    replacement: bool,
    num_workers: int,
    divisible_by_num_workers: bool,
) -> None:
    if not replacement:
        pytest.skip("replacement=False is not supported.")

    port = select_random_port()
    seed = 0
    world_size = 1

    torch.manual_seed(seed)

    epochs = 2
    sample_rate = 24000
    num_frames = 48000
    duration = 1
    input_keys = [
        "speech",
        "music",
        "effect",
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
        train_feature_dir = os.path.join(feature_dir, "tr")
        train_list_path = os.path.join(list_dir, "train.txt")

        os.makedirs(list_dir, exist_ok=True)
        os.makedirs(feature_dir, exist_ok=True)

        track_names = _save_dummy_dnr(
            root=feature_dir, sample_rate=sample_rate, num_frames=num_frames
        )
        train_track_names = track_names["train"]

        with open(train_list_path, mode="w") as f_list:
            for track_name in train_track_names:
                line = f"{track_name}\n"
                f_list.write(line)

        processes = []

        for rank in range(world_size):
            path = os.path.join(temp_dir, f"{rank}.pth")
            process = mp.Process(
                target=run_distributed_dnr_dataset_sampler,
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
            filenames_0 = torch.load(path, weights_only=True)

            assert len(filenames_0[epoch]) == expected_samples_per_epoch

            for rank in range(1, world_size):
                path = os.path.join(temp_dir, f"{rank}.pth")
                filenames_rank = torch.load(path, weights_only=True)

                # ensure disjointness among ranks
                assert filenames_0 != filenames_rank
                assert len(filenames_rank[epoch]) == expected_samples_per_epoch


def run_distributed_dnr_dataset_sampler(
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
    dataset = DistributedRandomStemsDNRDataset(
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


def _save_dummy_dnr(root: str, sample_rate: int, num_frames: int) -> Dict[str, List[int]]:
    g = torch.Generator()
    g.manual_seed(0)

    num_channels = 2

    train_track_names = v2_train_track_names[:10]
    validation_track_names = v2_validation_track_names[:10]
    test_track_names = v2_test_track_names[:10]

    track_names = {
        "train": train_track_names,
        "validation": validation_track_names,
        "test": test_track_names,
    }

    for subset_name, subset_track_names in zip(
        ["tr", "cv", "tt"], [train_track_names, validation_track_names, test_track_names]
    ):
        for track_name in subset_track_names:
            track_dir = os.path.join(root, subset_name, f"{track_name}")

            os.makedirs(track_dir, exist_ok=True)

            mixture = 0

            for source in sources:
                if source in ["speech", "music"]:
                    _source = source
                elif source == "effect":
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                path = os.path.join(track_dir, f"{_source}.wav")
                waveform = torch.randn((num_channels, num_frames), generator=g)
                max_amplitude = torch.max(torch.abs(waveform))
                waveform = 0.1 * (waveform / max_amplitude)
                mixture = mixture + waveform
                torchaudio.save(path, waveform, sample_rate)

            path = os.path.join(track_dir, "mix.wav")
            torchaudio.save(path, mixture, sample_rate)

    return track_names
