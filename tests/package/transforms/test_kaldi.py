import pytest
import torch
import torchaudio.compliance.kaldi as aCK
from dummy import allclose

from audyn.transforms.kaldi import KaldiMelSpectrogram, KaldiMFCC


@pytest.mark.parametrize("set_fbank_kwargs", [True, False])
def test_kaldi_melspectrogram(set_fbank_kwargs: bool) -> None:
    torch.manual_seed(0)

    sample_rate = 16000
    n_mels = 80
    frame_length = 25
    frame_shift = 10
    win_length = int(frame_length * sample_rate / 1000)
    hop_length = int(frame_shift * sample_rate / 1000)
    batch_size, in_channels, timesteps = 4, 1, 16000

    if set_fbank_kwargs:
        fbank_kwargs = {
            "sample_frequency": sample_rate,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "low_freq": 0,
            "high_freq": sample_rate / 2,
            "num_mel_bins": n_mels,
            "use_power": True,
        }
    else:
        fbank_kwargs = None

    # 2D input
    waveform = torch.randn((batch_size, timesteps))

    melspectrogram_transform = KaldiMelSpectrogram(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fbank_kwargs=fbank_kwargs,
    )
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:2] == (batch_size, n_mels)

    # 3D input
    waveform = torch.randn((batch_size, in_channels, timesteps))

    melspectrogram_transform = KaldiMelSpectrogram(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        fbank_kwargs=fbank_kwargs,
    )
    melspectrogram = melspectrogram_transform(waveform)

    assert melspectrogram.size()[:3] == (batch_size, in_channels, n_mels)

    # compatibility
    waveform = torch.randn((1, timesteps))

    melspectrogram_transform = KaldiMelSpectrogram(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    melspectrogram = melspectrogram_transform(waveform)

    melspectrogram_ack = aCK.fbank(
        waveform,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
    )

    allclose(melspectrogram, melspectrogram_ack.transpose(1, 0))


@pytest.mark.parametrize("set_mfcc_kwargs", [True, False])
def test_kaldi_mfcc(set_mfcc_kwargs: bool) -> None:
    torch.manual_seed(0)

    sample_rate = 16000
    n_mfcc = 20
    n_mels = 80
    frame_length = 25
    frame_shift = 10
    win_length = int(frame_length * sample_rate / 1000)
    hop_length = int(frame_shift * sample_rate / 1000)
    batch_size, in_channels, timesteps = 4, 1, 16000

    if set_mfcc_kwargs:
        mfcc_kwargs = {
            "sample_frequency": sample_rate,
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "low_freq": 0,
            "high_freq": sample_rate / 2,
            "num_mel_bins": n_mels,
            "use_energy": False,
        }
    else:
        mfcc_kwargs = None

    # 2D input
    waveform = torch.randn((batch_size, timesteps))

    mfcc_transform = KaldiMFCC(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        mfcc_kwargs=mfcc_kwargs,
    )
    mfcc = mfcc_transform(waveform)

    assert mfcc.size()[:2] == (batch_size, n_mfcc)

    # 3D input
    waveform = torch.randn((batch_size, in_channels, timesteps))

    mfcc_transform = KaldiMFCC(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
        mfcc_kwargs=mfcc_kwargs,
    )
    mfcc = mfcc_transform(waveform)

    assert mfcc.size()[:3] == (batch_size, in_channels, n_mfcc)

    # compatibility
    waveform = torch.randn((1, timesteps))

    mfcc_transform = KaldiMFCC(
        sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_mfcc=n_mfcc,
        n_mels=n_mels,
    )
    mfcc = mfcc_transform(waveform)

    mfcc_ack = aCK.mfcc(
        waveform,
        frame_length=frame_length,
        frame_shift=frame_shift,
        num_ceps=n_mfcc,
        num_mel_bins=n_mels,
        sample_frequency=sample_rate,
    )

    allclose(mfcc, mfcc_ack.transpose(1, 0))
