import os
import tempfile

import pytest
import torch
import torchaudio
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

from audyn.utils.data.dataloader import DistributedDataLoader
from audyn.utils.data.dataset import WebDatasetWrapper


def test_distributed_dataloader() -> None:
    batch_size = 2
    num_replicas = 2

    dataset = DummyDataset()

    dataloader_rank0 = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=0,
        shuffle=True,
    )
    dataloader_rank1 = DistributedDataLoader(
        dataset,
        batch_size=batch_size,
        num_replicas=num_replicas,
        rank=1,
        shuffle=True,
    )
    data_rank0 = []
    data_rank1 = []

    for data in dataloader_rank0:
        data = data.view(-1).tolist()
        data_rank0 = data_rank0 + data

    for data in dataloader_rank1:
        data = data.view(-1).tolist()
        data_rank1 = data_rank1 + data

    # should be disjoint
    assert set(data_rank0) & set(data_rank1) == set()


@pytest.mark.parametrize("decode_audio_as_waveform", [True, False])
@pytest.mark.parametrize("decode_audio_as_monoral", [True, False])
def test_dataloader_for_composer(decode_audio_as_waveform, decode_audio_as_monoral: bool) -> None:
    torch.manual_seed(0)

    max_shard_size = 5

    batch_size = 6
    sample_rate = 16000
    duration = 10
    num_channels = 2
    num_files = 2 * batch_size

    with tempfile.TemporaryDirectory() as temp_dir:
        # save .flac files
        audio_dir = os.path.join(temp_dir, "audio")
        feature_dir = os.path.join(temp_dir, "feature")
        list_path = os.path.join(temp_dir, "list.txt")

        os.makedirs(audio_dir)
        os.makedirs(feature_dir)

        with open(list_path, mode="w") as f:
            for idx in range(num_files):
                waveform = torch.randn((num_channels, int(sample_rate * duration)))
                path = os.path.join(audio_dir, f"{idx}.flac")
                torchaudio.save(path, waveform, sample_rate=sample_rate, format="flac")

                f.write(f"{idx}\n")

        template_path = os.path.join(feature_dir, "%02d.tar")

        with wds.ShardWriter(template_path, maxsize=max_shard_size) as sink:
            for idx in range(num_files):
                path = os.path.join(audio_dir, f"{idx}.flac")

                with open(path, mode="rb") as f:
                    audio = f.read()

                feature = {
                    "__key__": str(idx),
                    "filename.txt": str(idx),
                    "audio.flac": audio,
                }

                sink.write(feature)

        dataset = WebDatasetWrapper.instantiate_dataset(
            list_path,
            feature_dir,
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for batch in dataloader:
            assert len(batch["filename.txt"]) == batch_size

            audio = batch["audio.flac"]

            if decode_audio_as_waveform:
                waveform = audio
                audio_sample_rate = None
            else:
                waveform, audio_sample_rate = audio

            if decode_audio_as_monoral:
                assert waveform.size() == (
                    batch_size,
                    int(sample_rate * duration),
                )
            else:
                assert waveform.size() == (
                    batch_size,
                    num_channels,
                    int(sample_rate * duration),
                )

            if audio_sample_rate is not None:
                assert torch.all(audio_sample_rate == sample_rate)


class DummyDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor([idx])

    def __len__(self) -> int:
        return 10
