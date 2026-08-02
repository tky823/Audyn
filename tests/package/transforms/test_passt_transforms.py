import pytest
import torch

from audyn.transforms.passt import PaSSTMelSpectrogram


@pytest.mark.parametrize("take_log", [True, False])
def test_passt_melspectrogram(take_log: bool) -> None:
    torch.manual_seed(0)

    sample_rate = 32000
    duration = 20
    n_mels = 128
    freq_mask_param = 10
    time_mask_param = 20

    batch_size = 8
    timesteps = int(duration * sample_rate)

    melspectrogram_transform = PaSSTMelSpectrogram.build_from_default_config(
        "audioset",
        sample_rate,
        n_mels=n_mels,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
        take_log=take_log,
    )

    waveform = torch.randn((batch_size, timesteps))

    melspectrogram_transform.train()
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:2] == (batch_size, n_mels)

    melspectrogram_transform.eval()
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:2] == (batch_size, n_mels)
