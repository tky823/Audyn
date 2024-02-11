import pytest
import torch

from audyn.transforms.ast import SelfSupervisedAudioSpectrogramTransformerMelSpectrogram


@pytest.mark.parametrize("duration", [2.0, 4.0])
@pytest.mark.parametrize("n_frames", [None, 256])
@pytest.mark.parametrize("take_log", [True, False])
def test_ssast_melspectrogram(duration: float, n_frames: int, take_log: bool) -> None:
    torch.manual_seed(0)

    sample_rate = 16000
    n_mels = 128

    batch_size = 8
    timesteps = int(duration * sample_rate)

    melspectrogram_transform = SelfSupervisedAudioSpectrogramTransformerMelSpectrogram(
        sample_rate,
        n_mels=n_mels,
        n_frames=n_frames,
        take_log=take_log,
    )

    waveform = torch.randn((batch_size, timesteps))
    melspectrogram = melspectrogram_transform(waveform)

    if n_frames is not None:
        assert melspectrogram.size() == (batch_size, n_mels, n_frames)
    else:
        assert melspectrogram.size()[:2] == (batch_size, n_mels)
