import os
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.distributed as dist
import torchaudio
from torch.utils.data import Dataset, IterableDataset, RandomSampler, get_worker_info
from torchaudio.io import StreamReader

from .distributed import DistributedRandomStemsMUSDB18Sampler
from .sampler import RandomStemsMUSDB18Sampler

__all__ = [
    "MUSDB18",
    "StemsMUSDB18Dataset",
    "RandomStemsMUSDB18Dataset",
    "DistributedRandomStemsMUSDB18Dataset",
    "Track",
]


class MUSDB18(Dataset):
    """MUSDB18 dataset.

    Args:
        root (str): Root of MUSDB18 dataset.
        subset (str): ``train``, ``validation``, or ``test``.
        ext (str): Extension of audio files. ``wav`` and ``mp4`` are supported.
            Default: ``mp4``.

    .. note::

        We assume following structure when ``ext=mp4``

        .. code-block:: shell

            - root/  # typically MUSDB18, MUSDB18-HQ, MUSDB18-7s
                |- train/
                    |- A Classic Education - NightOwl.mp4
                    ...
                |- test/
                    ...

        We assume following structure when ``ext=wav``

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

    def __init__(self, root: str, subset: str, ext: str = "mp4") -> None:
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
        self.ext = ext

        self.track_names = self._validate_tracks(track_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_name = self.track_names[idx]
        track = Track(self.root, track_name, ext=self.ext)

        return track

    def __len__(self) -> int:
        return len(self.track_names)

    def _validate_tracks(self, track_names: List[str]) -> List[str]:
        from . import sources

        root = self.root

        subset_dir = "test" if self.subset == "test" else "train"
        existing_track_names = []

        for track_name in track_names:
            existing = True

            if self.ext in ["wav", ".wav"]:
                for source in ["mixture"] + sources:
                    filename = f"{track_name}/{source}.wav"
                    path = os.path.join(root, subset_dir, filename)

                    if not os.path.exists(path):
                        existing = False
                        warnings.warn(f"{path} is not found.", UserWarning, stacklevel=2)
            elif self.ext in ["mp4", "stem.mp4", ".mp4", ".stem.mp4"]:
                filename = f"{track_name}.stem.mp4"
                path = os.path.join(root, subset_dir, filename)

                if not os.path.exists(path):
                    existing = False
                    warnings.warn(f"{path} is not found.", UserWarning, stacklevel=2)
            else:
                raise ValueError(f"{self.ext} is not supported as extension.")

            if existing:
                existing_track_names.append(track_name)

        if len(existing_track_names) == 0:
            raise RuntimeError("There are no tracks.")

        return existing_track_names


class StemsMUSDB18Dataset(Dataset):
    """MUSDB18 dataset for evaluation.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .wav files.
        duration (float): Duration of waveform slice.
        drums_key (str): Key to store ``drums`` waveform.
        bass_key (str): Key to store ``bass`` waveform.
        other_key (str): Key to store ``other`` waveform.
        vocals_key (str): Key to store ``vocals`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        training (bool): If ``True``, segments are randomly selected.
            Otherwise, middle frames are selected.
        seed (int): Random seed for training.

    .. note::

        We assume following structure.

        .. code-block:: shell

            |- feature_dir/  # typically train or test
                |- A Classic Education - NightOwl/
                    |- mixture.wav
                    |- drums.wav
                    |- bass.wav
                    |- other.wav
                    |- vocals.wav
                ...

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        duration: float,
        drums_key: str = "drums",
        bass_key: str = "bass",
        other_key: str = "other",
        vocals_key: str = "vocals",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        training: bool = False,
        decode_audio_as_monoral: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__()

        filenames = []

        with open(list_path) as f:
            for line in f:
                filenames.append(line.strip("\n"))

        self.feature_dir = feature_dir
        self.filenames = filenames
        self.duration = duration
        self.source_keys = [
            drums_key,
            bass_key,
            other_key,
            vocals_key,
        ]
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.training = training
        self.decode_audio_as_monoral = decode_audio_as_monoral

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        self._validate_tracks()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        from . import sources

        feature_dir = self.feature_dir
        source_keys = self.source_keys
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        filename = self.filenames[idx]
        feature = {}

        for source, source_key in zip(sources, source_keys):
            filename_per_source = f"{filename}/{source}.wav"
            path = os.path.join(feature_dir, filename_per_source)
            waveform, sample_rate = self.load_sliced_audio(path)

            if sample_rate_key in feature:
                assert feature[sample_rate_key].item() == sample_rate
            else:
                feature[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)

            feature[source_key] = waveform

        feature[filename_key] = filename

        return feature

    def __len__(self) -> int:
        return len(self.filenames)

    def _validate_tracks(self) -> None:
        from . import sources

        feature_dir = self.feature_dir

        for filename in self.filenames:
            for source in ["mixture"] + sources:
                filename_per_source = f"{filename}/{source}.wav"
                path = os.path.join(feature_dir, filename_per_source)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")

    def load_sliced_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        duration = self.duration
        metadata = torchaudio.info(path)
        num_all_frames = metadata.num_frames
        num_frames = int(metadata.sample_rate * duration)

        if self.training:
            frame_offset = torch.randint(
                0, num_all_frames - num_frames, (), generator=self.generator
            )
            frame_offset = frame_offset.item()
        else:
            frame_offset = (num_all_frames - num_frames) // 2

        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames
        )

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        return waveform, sample_rate


class RandomStemsMUSDB18Dataset(IterableDataset):
    """MUSDB18 dataset for random mixing.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .wav files.
        duration (float): Duration of waveform slice.
        drums_key (str): Key to store ``drums`` waveform.
        bass_key (str): Key to store ``bass`` waveform.
        other_key (str): Key to store ``other`` waveform.
        vocals_key (str): Key to store ``vocals`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        replacement (bool): If ``True``, samples are taken with replacement.
        num_samples (int, optional): Number of samples per epoch. ``len(track_names)`` is
            used by default.
        seed (int): Random seed to set dataset and sampler state.
        generator (torch.Generator, optional): Random number generator to
            determine slicing positions of audio.

    .. note::

        We assume following structure.

        .. code-block:: shell

            |- feature_dir/  # typically train or test
                |- A Classic Education - NightOwl/
                    |- mixture.wav
                    |- drums.wav
                    |- bass.wav
                    |- other.wav
                    |- vocals.wav
                ...

    .. note::

        Internally, ``RandomStemsMUSDB18Sampler`` is used as sampler if ``replacement=True``.
        Otherwise, ``RandomSampler`` is used as sampler. Be careful of the difference in behavior
        of sampler depending on ``replacement``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        duration: float,
        drums_key: str = "drums",
        bass_key: str = "bass",
        other_key: str = "other",
        vocals_key: str = "vocals",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        replacement: bool = True,
        num_samples: Optional[int] = None,
        decode_audio_as_monoral: bool = False,
        seed: int = 0,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        filenames = []

        with open(list_path) as f:
            for line in f:
                filenames.append(line.strip("\n"))

        self.feature_dir = feature_dir
        self.filenames = filenames
        self.duration = duration
        self.seed = seed
        self.source_keys = [
            drums_key,
            bass_key,
            other_key,
            vocals_key,
        ]
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.decode_audio_as_monoral = decode_audio_as_monoral

        self.worker_id = None
        self.num_workers = None
        self.generator = None

        if num_samples is None:
            num_samples = len(filenames)

        if replacement:
            self.sampler = RandomStemsMUSDB18Sampler(
                filenames,
                num_samples=num_samples,
                generator=generator,
            )
        else:
            if num_samples > len(filenames):
                raise ValueError(
                    f"num_samples ({num_samples}) is greater than "
                    f"length of filenames ({len(filenames)})."
                )

            self.sampler = RandomSampler(
                filenames,
                replacement=replacement,
                num_samples=num_samples,
                generator=generator,
            )

        self.replacement = replacement
        self.num_total_samples = num_samples

        self._validate_tracks()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from . import sources

        feature_dir = self.feature_dir
        source_keys = self.source_keys
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        assert len(sources) == len(source_keys)

        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                self.num_workers = 1
            else:
                self.worker_id = worker_info.id
                self.num_workers = worker_info.num_workers

            # set generator state
            if self.replacement:
                # NOTE: Seed is dependent on worker_id,
                #       so random state of self.generator is not shared among processes.
                seed = self.seed + self.worker_id
            else:
                # NOTE: Seed is independent of worker_id,
                #       so random state of self.generator is shared among processes.
                seed = self.seed

            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

            # set sampler state
            sampler = self.sampler
            num_total_samples = self.num_total_samples

            if self.replacement:
                num_samples_per_worker = num_total_samples // self.num_workers

                if self.worker_id < num_total_samples % self.num_workers:
                    num_samples_per_worker += 1

                self.sampler = RandomStemsMUSDB18Sampler(
                    self.filenames,
                    num_samples=num_samples_per_worker,
                    generator=sampler.generator,
                )
            else:
                self.sampler = RandomSampler(
                    self.filenames,
                    replacement=self.replacement,
                    num_samples=num_total_samples,
                    generator=sampler.generator,
                )

        if self.replacement:
            filenames_per_worker = self.filenames
        else:
            # If self.replacement=False, track names should be disjointed among workers.
            filenames = self.filenames
            sampler = self.sampler
            num_total_samples = self.num_total_samples
            num_samples_per_worker = num_total_samples // self.num_workers

            if self.worker_id < num_total_samples % self.num_workers:
                num_samples_per_worker += 1

            # NOTE: Random state of self.generator is shared among processes.
            #       Random state of sampler.generator is not shared among processes.
            indices = torch.randperm(len(filenames), generator=self.generator)
            indices = indices.tolist()
            filenames_per_worker = [
                filenames[idx] for idx in indices[self.worker_id :: self.num_workers]
            ]
            self.sampler = RandomSampler(
                filenames_per_worker,
                replacement=self.replacement,
                num_samples=num_samples_per_worker,
                generator=sampler.generator,
            )

        for indices in self.sampler:
            filenames = []
            feature = {}

            if self.replacement:
                # If self.replacement=True,
                # indices returned by sampler should contain multiple indices.
                assert len(indices) == len(sources)
            else:
                # If self.replacement=False,
                # indices returned by sampler should contain single index.
                indices = [indices] * len(sources)

            for idx, source, source_key in zip(indices, sources, source_keys):
                filename = filenames_per_worker[idx]
                filenames.append(filename)
                filename_per_source = f"{filename}/{source}.wav"
                path = os.path.join(feature_dir, filename_per_source)
                waveform, sample_rate = self.load_randomly_sliced_audio(path)

                if sample_rate_key in feature:
                    assert feature[sample_rate_key].item() == sample_rate
                else:
                    feature[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)

                feature[source_key] = waveform

            if self.replacement:
                feature[filename_key] = "+".join(filenames)
            else:
                assert len(set(filenames)) == 1, (
                    "Even when self.sampler.replacement=False, "
                    "different tracks are chosen among sources."
                )

                feature[filename_key] = filenames[0]

            yield feature

    def __len__(self) -> int:
        return self.num_total_samples

    def _validate_tracks(self) -> None:
        from . import sources

        feature_dir = self.feature_dir

        for filename in self.filenames:
            for source in ["mixture"] + sources:
                filename_per_source = f"{filename}/{source}.wav"
                path = os.path.join(feature_dir, filename_per_source)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")

    def load_randomly_sliced_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        duration = self.duration
        metadata = torchaudio.info(path)
        num_all_frames = metadata.num_frames
        num_frames = int(metadata.sample_rate * duration)
        frame_offset = torch.randint(0, num_all_frames - num_frames, (), generator=self.generator)
        frame_offset = frame_offset.item()
        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames
        )

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        return waveform, sample_rate


class DistributedRandomStemsMUSDB18Dataset(IterableDataset):
    """MUSDB18 dataset for random mixing with distributed training.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .wav files.
        duration (float): Duration of waveform slice.
        drums_key (str): Key to store ``drums`` waveform.
        bass_key (str): Key to store ``bass`` waveform.
        other_key (str): Key to store ``other`` waveform.
        vocals_key (str): Key to store ``vocals`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        replacement (bool): If ``True``, samples are taken with replacement.
        num_samples (int, optional): Number of samples per epoch.
            ``len(track_names)`` is used by default.
        seed (int): Random seed to set dataset and sampler state.
        generator (torch.Generator, optional): Random number generator to
            determine slicing positions of audio.

    .. note::

        We assume following structure.

        .. code-block:: shell

            |- feature_dir/  # typically train or test
                |- A Classic Education - NightOwl/
                    |- mixture.wav
                    |- drums.wav
                    |- bass.wav
                    |- other.wav
                    |- vocals.wav
                ...

    .. note::

        Internally, ``RandomStemsMUSDB18Sampler`` is used as sampler if ``replacement=True``.
        Otherwise, ``RandomSampler`` is used as sampler. Be careful of the difference in behavior
        of sampler depending on ``replacement``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        duration: float,
        drums_key: str = "drums",
        bass_key: str = "bass",
        other_key: str = "other",
        vocals_key: str = "vocals",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        replacement: bool = True,
        num_samples: Optional[int] = None,
        decode_audio_as_monoral: bool = False,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        drop_last: Optional[bool] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        super().__init__()

        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()

        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        if drop_last is None:
            if replacement:
                drop_last = False
            else:
                drop_last = True

        filenames = []

        with open(list_path) as f:
            for line in f:
                filenames.append(line.strip("\n"))

        self.feature_dir = feature_dir
        self.filenames = filenames
        self.duration = duration
        self.seed = seed
        self.source_keys = [
            drums_key,
            bass_key,
            other_key,
            vocals_key,
        ]
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.decode_audio_as_monoral = decode_audio_as_monoral

        self.num_replicas = num_replicas
        self.rank = rank
        self.worker_id = None
        self.num_workers = None
        self.generator = None

        if num_samples is None:
            num_samples = len(filenames)

        num_samples_per_replica = num_samples // num_replicas

        if num_samples % num_replicas > 0 and not drop_last:
            num_samples_per_replica += 1

        if replacement:
            self.sampler = DistributedRandomStemsMUSDB18Sampler(
                filenames,
                num_samples=num_samples_per_replica,
                num_replicas=num_replicas,
                rank=rank,
                seed=seed,
            )
        else:
            if num_replicas * num_samples_per_replica > len(filenames):
                raise ValueError(
                    f"num_samples ({num_samples}) is greater than "
                    f"length of filenames ({len(filenames)})."
                    " You may need to set drop_last=True."
                )

            self.sampler = RandomSampler(
                filenames,
                replacement=replacement,
                num_samples=num_samples_per_replica,
                generator=generator,
            )

        self.replacement = replacement
        self.num_total_samples = num_samples
        self.num_samples_per_replica = num_samples_per_replica

        self._validate_tracks()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        from . import sources

        num_replicas = self.num_replicas
        rank = self.rank

        feature_dir = self.feature_dir
        source_keys = self.source_keys
        sample_rate_key = self.sample_rate_key
        filename_key = self.filename_key

        assert len(sources) == len(source_keys)

        if self.worker_id is None:
            # should be initialized
            worker_info = get_worker_info()

            if worker_info is None:
                self.worker_id = 0
                self.num_workers = 1
            else:
                self.worker_id = worker_info.id
                self.num_workers = worker_info.num_workers

            # set generator state
            if self.replacement:
                # NOTE: Seed is dependent on worker_id,
                #       so random state of self.generator is not shared among processes.
                seed = self.seed + self.num_workers * rank + self.worker_id
            else:
                # NOTE: Seed is independent of worker_id,
                #       so random state of self.generator is shared among processes.
                seed = self.seed

            self.generator = torch.Generator()
            self.generator.manual_seed(seed)

            # set sampler state
            sampler = self.sampler
            num_samples_per_replica = self.num_samples_per_replica

            if self.replacement:
                num_samples_per_worker = num_samples_per_replica // self.num_workers

                if self.worker_id < num_samples_per_replica % self.num_workers:
                    num_samples_per_worker += 1

                self.sampler = DistributedRandomStemsMUSDB18Sampler(
                    self.filenames,
                    num_samples=num_samples_per_worker,
                    num_replicas=num_replicas,
                    rank=rank,
                    seed=sampler.seed,
                )
            else:
                self.sampler = RandomSampler(
                    self.filenames,
                    replacement=self.replacement,
                    num_samples=num_samples_per_replica,
                    generator=sampler.generator,
                )

        if self.replacement:
            filenames_per_worker = self.filenames
        else:
            # If self.replacement=False, track names should be disjointed among ranks and workers.
            sampler = self.sampler
            num_samples_per_replica = self.num_samples_per_replica
            num_samples_per_worker = num_samples_per_replica // self.num_workers

            if self.worker_id < num_samples_per_replica % self.num_workers:
                num_samples_per_worker += 1

            # NOTE: Random state of self.generator is shared among processes.
            #       Random state of sampler.generator is not shared among processes.
            indices = torch.randperm(len(self.filenames), generator=self.generator)
            indices = indices.tolist()
            offset = self.num_workers * rank + self.worker_id
            step = num_replicas * self.num_workers
            filenames_per_worker = [self.filenames[idx] for idx in indices[offset::step]]
            self.sampler = RandomSampler(
                filenames_per_worker,
                replacement=self.replacement,
                num_samples=num_samples_per_worker,
                generator=sampler.generator,
            )

        for indices in self.sampler:
            filenames = []
            feature = {}

            if self.replacement:
                # If self.replacement=True,
                # indices returned by sampler should contain multiple indices.
                assert len(indices) == len(sources)
            else:
                # If self.replacement=False,
                # indices returned by sampler should contain single index.
                indices = [indices] * len(sources)

            for idx, source, source_key in zip(indices, sources, source_keys):
                filename = filenames_per_worker[idx]
                filenames.append(filename)
                filename_per_source = f"{filename}/{source}.wav"
                path = os.path.join(feature_dir, filename_per_source)
                waveform, sample_rate = self.load_randomly_sliced_audio(path)

                if sample_rate_key in feature:
                    assert feature[sample_rate_key].item() == sample_rate
                else:
                    feature[sample_rate_key] = torch.tensor(sample_rate, dtype=torch.long)

                feature[source_key] = waveform

            if self.replacement:
                feature[filename_key] = "+".join(filenames)
            else:
                assert len(set(filenames)) == 1, (
                    "Even when self.sampler.replacement=False, "
                    "different tracks are chosen among sources."
                )

                feature[filename_key] = filenames[0]

            yield feature

    def __len__(self) -> int:
        return self.num_samples_per_replica

    def _validate_tracks(self) -> None:
        from . import sources

        feature_dir = self.feature_dir

        for filename in self.filenames:
            for source in ["mixture"] + sources:
                filename_per_source = f"{filename}/{source}.wav"
                path = os.path.join(feature_dir, filename_per_source)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")

    def load_randomly_sliced_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        duration = self.duration
        metadata = torchaudio.info(path)
        num_all_frames = metadata.num_frames
        num_frames = int(metadata.sample_rate * duration)
        frame_offset = torch.randint(0, num_all_frames - num_frames, (), generator=self.generator)
        frame_offset = frame_offset.item()
        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames
        )

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        return waveform, sample_rate


class Track:
    """Music track for MUSDB18 dataset.

    Args:
        root (str): Root directory of dataset.
        name (str): Track name. e.g. ``A Classic Education - NightOwl``.
        ext (str): Extension to load file. ``mp4`` and ``wav`` are supported.
            Default: ``mp4``.

    """

    def __init__(
        self,
        root: str,
        name: str,
        ext: str = "mp4",
    ) -> None:
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
        self._ext = ext

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

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            instrument="mixture",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
            ext=self._ext,
        )

        return waveform, sample_rate

    @property
    def drums(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            instrument="drums",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
            ext=self._ext,
        )

        return waveform, sample_rate

    @property
    def bass(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            instrument="bass",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
            ext=self._ext,
        )

        return waveform, sample_rate

    @property
    def other(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            instrument="other",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
            ext=self._ext,
        )

        return waveform, sample_rate

    @property
    def vocals(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            instrument="vocals",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
            ext=self._ext,
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
            _waveform, _sample_rate = _load_single_stream(
                root,
                subset,
                name,
                instrument=instrument,
                frame_offset=self.frame_offset,
                num_frames=self.num_frames,
                ext=self._ext,
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
            _waveform, _sample_rate = _load_single_stream(
                root,
                subset,
                name,
                instrument=source,
                frame_offset=self.frame_offset,
                num_frames=self.num_frames,
                ext=self._ext,
            )

            if sample_rate is None:
                sample_rate = _sample_rate
            else:
                assert _sample_rate == sample_rate

            waveform.append(_waveform)

        waveform = torch.stack(waveform, dim=0)

        return waveform, sample_rate


def _load_single_stream(
    root: str,
    subset: str,
    name: str,
    instrument: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    ext: str = "mp4",
) -> Tuple[torch.Tensor, int]:
    from . import sources

    streams = ["mixture"] + sources

    if ext in ["wav", ".wav"]:
        filename = f"{name}/{instrument}.wav"
        path = os.path.join(root, subset, filename)

        waveform, sample_rate = torchaudio.load(
            path,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
    elif ext in ["mp4", ".mp4", "stem.mp4", ".stem.mp4"]:
        filename = f"{name}.stem.mp4"
        stream_idx = streams.index(instrument)
        path = os.path.join(root, subset, filename)
        reader = StreamReader(path)
        stream_info = reader.get_src_stream_info(stream_idx)
        sample_rate = stream_info.sample_rate
        timestamp = frame_offset / sample_rate

        assert sample_rate == 44100

        sample_rate = int(sample_rate)

        if num_frames < 0:
            reader.add_audio_stream(
                frames_per_chunk=sample_rate,
                stream_index=stream_idx,
            )

            waveform = []

            for (chunk,) in reader.stream():
                waveform.append(chunk)

            waveform = torch.cat(waveform, dim=0)
        else:
            reader.add_audio_stream(
                frames_per_chunk=sample_rate,
                stream_index=stream_idx,
            )
            reader.seek(timestamp)
            reader.fill_buffer()
            (waveform,) = reader.pop_chunks()

        waveform = waveform.permute(1, 0)
        waveform = waveform.contiguous()
    else:
        raise ValueError(f"{ext} is not supported as extension.")

    return waveform, sample_rate
