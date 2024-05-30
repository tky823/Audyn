from .._common.dataset import BirdCLEFPrimaryLabelDataset

__all__ = [
    "BirdCLEF2021PrimaryLabelDataset",
]


class BirdCLEF2021PrimaryLabelDataset(BirdCLEFPrimaryLabelDataset):
    """Dataset for training of bird classification model in BirdCLEF2021.

    Args:
        list_path (str): Path to list file. Each entry represents path to audio file
            without extension such as ``abethr1/XC128013``.
        feature_dir (str): Path to dataset containing ``train_metadata.csv`` file,
            ``train_audio`` directory, and so on.
        audio_key (str): Key of audio.
        sample_rate_key (str): Key of sampling rate.
        label_name_key (str): Key of prmary label name in given sample.
        filename_key (str): Key of filename in given sample.
        decode_audio_as_waveform (bool, optional): If ``True``, audio is decoded as waveform
            tensor and sampling rate is ignored. Otherwise, audio is decoded as tuple of
            waveform tensor and sampling rate. Default: ``True``.
        decode_audio_as_monoral (bool, optional): If ``True``, decoded audio is treated as
            monoral waveform of shape (num_samples,) by reducing channel dimension. Otherwise,
            shape of waveform is (num_channels, num_samples), which is returned by
            ``torchaudio.load``. Default: ``True``.

    """
