import torchaudio
from packaging import version

IS_TORCHAUDIO_LT_2_9 = version.parse(torchaudio.__version__) < version.parse("2.9")

__all__ = ["AudioMetadata", "info"]


class AudioMetadata:
    def __init__(self, path: str) -> None:
        if IS_TORCHAUDIO_LT_2_9:
            self._meatadata = torchaudio.info(path)
            self._decoder = None
        else:
            from torchcodec.decoders import AudioDecoder

            self._meatadata = None
            self._decoder = AudioDecoder(path)

    @property
    def sample_rate(self) -> int:
        if IS_TORCHAUDIO_LT_2_9:
            return self._meatadata.sample_rate
        else:
            return self._decoder.metadata.sample_rate

    @property
    def num_channels(self) -> int:
        if IS_TORCHAUDIO_LT_2_9:
            return self._meatadata.num_channels
        else:
            return self._decoder.metadata.num_channels

    @property
    def num_frames(self) -> int:
        if IS_TORCHAUDIO_LT_2_9:
            return self._meatadata.num_frames
        else:
            return int(self._decoder.metadata.duration_seconds_from_header * self.sample_rate)


def info(path: str, **kwargs) -> AudioMetadata:
    """Wrapper function of torchaudio.info."""
    return AudioMetadata(path, **kwargs)
