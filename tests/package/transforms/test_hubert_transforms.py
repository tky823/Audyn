import torch

from audyn.transforms.hubert import HuBERTMFCC


def test_hubert_mfcc() -> None:
    torch.manual_seed(0)

    sample_rate = 16000
    duration = 10
    timestep = int(sample_rate * duration)

    waveform = torch.randn((timestep,))
    mfcc_transform = HuBERTMFCC(sample_rate)
    mfcc = mfcc_transform(waveform)

    assert mfcc.size() == (3 * 13, 998)
