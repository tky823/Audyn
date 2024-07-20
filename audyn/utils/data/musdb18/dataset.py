import os
from typing import Any, Dict, Iterator, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .sampler import RandomStemsMUSDB18Sampler


class MUSDB18(Dataset):
    """MUSDB18 dataset.

    Args:
        root (str): Root of MUSDB18 dataset.
        subset (str): ``train``, ``validation``, or ``test``.

    .. note::

        We assume following structure.

        .. code-block:: shell

            - root/  # typically MUSDB18, MUSDB18-HQ, MUSDB18-7s
                |- train/
                    |- A Classic Education - NightOwl/
                        |- mixture.wav
                        |- drums.wav
                        |- bass.wav
                        |- other.wav
                        |- vocals.wav
                    ...
                |- test/
                    ...

    """

    def __init__(self, root: str, subset: str) -> None:
        from . import test_track_names, train_track_names, validation_track_names

        super().__init__()

        if subset == "train":
            track_names = train_track_names
        elif subset == "validation":
            track_names = validation_track_names
        elif subset == "test":
            track_names = test_track_names
        else:
            raise ValueError(f"{subset} is not supported as subset.")

        self.root = root
        self.subset = subset
        self.track_names = track_names

        self._validate_tracks()

    def __iter__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_name = self.track_names[idx]
        track = Track(self.root, track_name)

        return track

    def __len__(self) -> int:
        return len(self.track_names)

    def _validate_tracks(self) -> None:
        from . import sources

        root = self.root

        subset_dir = "test" if self.subset == "test" else "train"

        for track_name in self.track_names:
            for source in ["mixture"] + sources:
                filename = f"{track_name}/{source}.wav"
                path = os.path.join(root, subset_dir, filename)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")


class RandomStemsMUSDB18Dataset(IterableDataset):
    """MUSDB18 dataset for random mixing.

    Args:
        root (str): Root of MUSDB18 dataset.
        subset (str): ``train``, ``validation``, or ``test``.
        duration (float): Duration of waveform slice.
        drums_key (str): Key to store ``drums`` waveform.
        bass_key (str): Key to store ``bass`` waveform.
        other_key (str): Key to store ``other`` waveform.
        vocals_key (str): Key to store ``vocals`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        seed (int): Random seed to set sampler state.

    .. note::

        We assume following structure.

        .. code-block:: shell

            - root/  # typically MUSDB18, MUSDB18-HQ, MUSDB18-7s
                |- train/
                    |- A Classic Education - NightOwl/
                        |- mixture.wav
                        |- drums.wav
                        |- bass.wav
                        |- other.wav
                        |- vocals.wav
                    ...
                |- test/
                    ...

    """

    def __init__(
        self,
        root: str,
        subset: str,
        duration: float,
        drums_key: str = "drums",
        bass_key: str = "bass",
        other_key: str = "other",
        vocals_key: str = "vocals",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        seed: int = 0,
    ) -> None:
        from . import test_track_names, train_track_names, validation_track_names

        super().__init__()

        if subset == "train":
            track_names = train_track_names
        elif subset == "validation":
            track_names = validation_track_names
        elif subset == "test":
            track_names = test_track_names
        else:
            raise ValueError(f"{subset} is not supported as subset.")

        self.root = root
        self.subset = subset
        self.track_names = track_names
        self.duration = duration
        self.seed = seed
        self.source_keys = [drums_key, bass_key, other_key, vocals_key]
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key

        self.worker_id = None
        self.generator = None
        self.sampler = RandomStemsMUSDB18Sampler(track_names)

        self._validate_tracks()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from . import sources

        root = self.root
        source_keys = self.source_keys
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        subset_dir = "test" if self.subset == "test" else "train"

        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                num_workers = 1
            else:
                self.worker_id = worker_info.id
                num_workers = worker_info.num_workers

            # set generator state
            seed = self.seed + self.worker_id
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

            # set sampler state
            sampler = self.sampler
            num_total_samples = sampler.num_samples // len(sources)
            num_samples_per_worker = num_total_samples // num_workers

            if self.worker_id < num_total_samples % num_workers:
                num_samples_per_worker += 1

            self.sampler = RandomStemsMUSDB18Sampler(
                sampler.track_names,
                replacement=sampler.replacement,
                num_samples=num_samples_per_worker,
            )

        for indices in self.sampler:
            track_names = []
            feature = {}

            assert len(indices) == len(sources) == len(source_keys)

            for idx, source, source_key in zip(indices, sources, source_keys):
                track_name = self.track_names[idx]
                track_names.append(track_name)
                filename = f"{track_name}/{source}.wav"
                path = os.path.join(root, subset_dir, filename)
                waveform, sample_rate = self.load_sliced_audio(path)

                if sample_rate_key in feature:
                    assert feature[sample_rate_key].item() == sample_rate
                else:
                    feature[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)

                feature[source_key] = waveform

            feature[filename_key] = "+".join(track_names)

            yield feature

    def __len__(self) -> int:
        return len(self.track_names)

    def _validate_tracks(self) -> None:
        from . import sources

        root = self.root

        subset_dir = "test" if self.subset == "test" else "train"

        for track_name in self.track_names:
            for source in ["mixture"] + sources:
                filename = f"{track_name}/{source}.wav"
                path = os.path.join(root, subset_dir, filename)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")

    def load_sliced_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        duration = self.duration
        metadata = torchaudio.info(path)
        num_all_frames = metadata.num_frames
        num_frames = int(metadata.sample_rate * duration)
        frame_offset = torch.randint(0, num_all_frames - num_frames, (), generator=self.generator)
        frame_offset = frame_offset.item()
        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames
        )

        return waveform, sample_rate


class Track:
    def __init__(self, root: str, name: str) -> None:
        from . import test_track_names, train_track_names, validation_track_names

        if name in train_track_names:
            _subset = "train"
        elif name in validation_track_names:
            _subset = "validation"
        elif name in test_track_names:
            _subset = "test"
        else:
            raise ValueError(f"{name} is not found.")

        self._root = root
        self._subset = _subset
        self._name = name

        self._frame_offset = 0
        self._num_frames = -1

    @property
    def name(self) -> str:
        return self._name

    @property
    def frame_offset(self) -> int:
        return self._frame_offset

    @frame_offset.setter
    def frame_offset(self, __value: int) -> None:
        self._frame_offset = __value

    @property
    def num_frames(self) -> int:
        return self._num_frames

    @num_frames.setter
    def num_frames(self, __value: int) -> None:
        self._num_frames = __value

    @property
    def mixture(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name
        filename = f"{name}/mixture.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def drums(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name
        filename = f"{name}/drums.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def bass(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name
        filename = f"{name}/bass.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def other(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name
        filename = f"{name}/other.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def vocals(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name
        filename = f"{name}/vocals.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def accompaniment(self) -> Tuple[torch.Tensor, int]:
        from . import accompaniments

        root = self._root
        subset = self._subset
        name = self.name

        waveform = 0
        sample_rate = None

        for instrument in accompaniments:
            filename = f"{name}/{instrument}.wav"
            path = os.path.join(root, subset, filename)

            _waveform, _sample_rate = torchaudio.load(
                path,
                frame_offset=self.frame_offset,
                num_frames=self.num_frames,
            )

            if sample_rate is None:
                sample_rate = _sample_rate
            else:
                assert _sample_rate == sample_rate

            waveform = waveform + _waveform

        return waveform, sample_rate

    @property
    def stems(self) -> Tuple[torch.Tensor, int]:
        from . import sources

        root = self._root
        subset = self._subset
        name = self.name

        waveform = []
        sample_rate = None

        for source in ["mixture"] + sources:
            filename = f"{name}/{source}.wav"
            path = os.path.join(root, subset, filename)

            _waveform, _sample_rate = torchaudio.load(
                path, frame_offset=self.frame_offset, num_frames=self.num_frames
            )

            if sample_rate is None:
                sample_rate = _sample_rate
            else:
                assert _sample_rate == sample_rate

            waveform.append(_waveform)

        waveform = torch.stack(waveform, dim=0)

        return waveform, sample_rate
