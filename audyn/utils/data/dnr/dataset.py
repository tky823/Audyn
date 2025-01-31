import os
import warnings
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset, IterableDataset, RandomSampler, get_worker_info

from .sampler import RandomStemsDNRSampler

__all__ = [
    "DNRDataset",
    "StemsDNRDataset",
    "RandomStemsDNRDataset",
    "Track",
]


class DNRDataset(Dataset):
    """DnR dataset.

    Args:
        root (str): Root of DnR dataset.
        subset (str): ``train``, ``validation``, or ``test``.

    .. note::

        We assume following structure

        .. code-block:: shell

            - root/  # dnr_v2
                |- tr/
                    |- 1002/
                    |- 10031/
                    |- 10032/
                    ...
                |- cv/
                    ...
                |- tt/

    """

    def __init__(self, root: str, subset: str, version: Union[int, str] = 2) -> None:
        from . import (
            v2_test_track_names,
            v2_train_track_names,
            v2_validation_track_names,
        )

        super().__init__()

        _version = str(version)

        assert _version == "2", "Only v2 is supported."

        if subset == "train":
            track_names = v2_train_track_names
        elif subset == "validation":
            track_names = v2_validation_track_names
        elif subset == "test":
            track_names = v2_test_track_names
        else:
            raise ValueError(f"{subset} is not supported as subset.")

        self.root = root
        self.subset = subset

        self.track_names = self._validate_tracks(track_names)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        track_name = self.track_names[idx]
        track = Track(self.root, track_name)

        return track

    def __len__(self) -> int:
        return len(self.track_names)

    def _validate_tracks(self, track_names: List[str]) -> List[str]:
        from . import sources

        root = self.root
        subset = self.subset

        if subset == "train":
            subset_dir = "tr"
        elif subset == "validation":
            subset_dir = "cv"
        elif subset == "test":
            subset_dir = "tt"
        else:
            raise ValueError(f"Invalid subset {subset} is given.")

        existing_track_names = []

        for track_name in track_names:
            existing = True

            for source in ["mixture"] + sources:
                if source in ["speech", "music"]:
                    _source = source
                elif source in ["mixture"]:
                    _source = "mix"
                elif source in ["effect", "sfx"]:
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                filename = f"{track_name}/{_source}.wav"
                path = os.path.join(root, subset_dir, filename)

                if not os.path.exists(path):
                    existing = False
                    warnings.warn(f"{path} is not found.", UserWarning, stacklevel=2)

            if existing:
                existing_track_names.append(track_name)

        if len(existing_track_names) == 0:
            raise RuntimeError("There are no tracks.")

        return existing_track_names


class StemsDNRDataset(Dataset):
    """DnR dataset for evaluation.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .wav files.
        duration (float): Duration of waveform slice.
        speech_key (str): Key to store ``speech`` waveform.
        music_key (str): Key to store ``music`` waveform.
        effect_key (str): Key to store ``effect`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.

    .. note::

        We assume following structure.

        .. code-block:: shell

            |- feature_dir/  # typically train or test
                |- 1002/
                    |- speech.wav
                    |- music.wav
                    |- sfx.wav
                ...

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        duration: float,
        speech_key: str = "speech",
        music_key: str = "music",
        effect_key: str = "effect",
        sample_rate_key: str = "sample_rate",
        filename_key: str = "filename",
        decode_audio_as_monoral: bool = False,
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
            speech_key,
            music_key,
            effect_key,
        ]
        self.sample_rate_key = sample_rate_key
        self.filename_key = filename_key
        self.decode_audio_as_monoral = decode_audio_as_monoral

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
            if source in ["speech", "music"]:
                _source = source
            elif source in ["mixture"]:
                _source = "mix"
            elif source in ["effect", "sfx"]:
                _source = "sfx"
            else:
                raise ValueError(f"Invalid source {source} is given.")

            filename_per_source = f"{filename}/{_source}.wav"
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
                if source in ["speech", "music"]:
                    _source = source
                elif source in ["mixture"]:
                    _source = "mix"
                elif source in ["effect", "sfx"]:
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                filename_per_source = f"{filename}/{_source}.wav"
                path = os.path.join(feature_dir, filename_per_source)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")

    def load_sliced_audio(self, path: str) -> Tuple[torch.Tensor, int]:
        duration = self.duration
        metadata = torchaudio.info(path)
        num_all_frames = metadata.num_frames
        num_frames = int(metadata.sample_rate * duration)
        frame_offset = (num_all_frames - num_frames) // 2
        waveform, sample_rate = torchaudio.load(
            path, frame_offset=frame_offset, num_frames=num_frames
        )

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        return waveform, sample_rate


class RandomStemsDNRDataset(IterableDataset):
    """DnR dataset for random mixing.

    Args:
        list_path (str): Path to list file containing filenames.
        feature_dir (str): Path to directory containing .wav files.
        duration (float): Duration of waveform slice.
        speech_key (str): Key to store ``speech`` waveform.
        music_key (str): Key to store ``music`` waveform.
        effect_key (str): Key to store ``effect`` waveform.
        sample_rate_key (str): Key to store sampling rate.
        filename_key (str): Key to store filename.
        replacement (bool): If ``True``, samples are taken with replacement.
        num_samples (int, optional): Number of sampler per epoch. ``len(track_names)`` is
            used by default.
        seed (int): Random seed to set dataset and sampler state.
        generator (torch.Generator, optional): Random number generator to
            determine slicing positions of audio.

    .. note::

        We assume following structure.

        .. code-block:: shell

            |- feature_dir/  # typically train or test
                |- 1002/
                    |- mix.wav
                    |- speech.wav
                    |- music.wav
                    |- sfx.wav
                ...


    .. note::

        Internally, ``RandomStemsDNRSampler`` is used as sampler if ``replacement=True``.
        Otherwise, ``RandomSampler`` is used as sampler. Be careful of the difference in behavior
        of sampler depending on ``replacement``.

    """

    def __init__(
        self,
        list_path: str,
        feature_dir: str,
        duration: float,
        speech_key: str = "speech",
        music_key: str = "music",
        effect_key: str = "effect",
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
            speech_key,
            music_key,
            effect_key,
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
            self.sampler = RandomStemsDNRSampler(
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

                self.sampler = RandomStemsDNRSampler(
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
            sampler = self.sampler
            num_total_samples = self.num_total_samples
            num_samples_per_worker = num_total_samples // self.num_workers

            if self.worker_id < num_total_samples % self.num_workers:
                num_samples_per_worker += 1

            # NOTE: Random state of self.generator is shared among processes.
            #       Random state of sampler.generator is not shared among processes.
            indices = torch.randperm(num_total_samples, generator=self.generator).tolist()
            filenames_per_worker = [
                self.filenames[idx] for idx in indices[self.worker_id :: self.num_workers]
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

            assert len(sources) == len(source_keys)

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

                if source in ["speech", "music"]:
                    _source = source
                elif source in ["mixture"]:
                    _source = "mix"
                elif source in ["effect", "sfx"]:
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                filename_per_source = f"{filename}/{_source}.wav"
                path = os.path.join(feature_dir, filename_per_source)
                waveform, sample_rate = self.load_sliced_audio(path)

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
                if source in ["speech", "music"]:
                    _source = source
                elif source in ["mixture"]:
                    _source = "mix"
                elif source in ["effect", "sfx"]:
                    _source = "sfx"
                else:
                    raise ValueError(f"Invalid source {source} is given.")

                filename_per_source = f"{filename}/{_source}.wav"
                path = os.path.join(feature_dir, filename_per_source)

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

        if self.decode_audio_as_monoral:
            waveform = waveform.mean(dim=0)

        return waveform, sample_rate


class Track:
    """Track for DnR dataset.

    Args:
        root (str): Root directory of dataset.
        name (int or str): Track name. e.g. ``1002``.

    """

    def __init__(
        self,
        root: str,
        name: Union[int, str],
    ) -> None:
        from . import (
            v2_test_track_names,
            v2_train_track_names,
            v2_validation_track_names,
        )

        _name = int(name)

        if _name in v2_train_track_names:
            _subset = "train"
        elif _name in v2_validation_track_names:
            _subset = "validation"
        elif _name in v2_test_track_names:
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

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            source="mixture",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def speech(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            source="speech",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def music(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            source="music",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

        return waveform, sample_rate

    @property
    def effect(self) -> Tuple[torch.Tensor, int]:
        root = self._root
        subset = self._subset
        name = self.name

        waveform, sample_rate = _load_single_stream(
            root,
            subset,
            name,
            source="effect",
            frame_offset=self.frame_offset,
            num_frames=self.num_frames,
        )

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
                source=source,
                frame_offset=self.frame_offset,
                num_frames=self.num_frames,
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
    name: Union[int, str],
    source: str,
    frame_offset: int = 0,
    num_frames: int = -1,
) -> Tuple[torch.Tensor, int]:
    if subset == "train":
        subset_dir = "tr"
    elif subset == "validation":
        subset_dir = "cv"
    elif subset == "test":
        subset_dir = "tt"
    else:
        raise ValueError(f"Invalid subset {subset} is given.")

    if source in ["speech", "music"]:
        _source = source
    elif source in ["mixture"]:
        _source = "mix"
    elif source in ["effect", "sfx"]:
        _source = "sfx"
    else:
        raise ValueError(f"Invalid source {source} is given.")

    filename = f"{name}/{_source}.wav"
    path = os.path.join(root, subset_dir, filename)

    waveform, sample_rate = torchaudio.load(
        path,
        frame_offset=frame_offset,
        num_frames=num_frames,
    )

    return waveform, sample_rate
