import os
from typing import Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


class MUSDB18(Dataset):
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

    def __getitem__(self, idx: int) -> "Track":
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
            for source in sources + ["mixture"]:
                filename = f"{track_name}/{source}.wav"
                path = os.path.join(root, subset_dir, filename)

                if not os.path.exists(path):
                    raise FileNotFoundError(f"{path} is not found.")


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
