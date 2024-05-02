import re
from typing import Any, Dict, Iterable

from .webdataset import decode_audio, supported_audio_extensions

__all__ = ["Composer"]


class Composer:
    """Composer given to process each sample in list of samples.

    This class is mainly used for webdataset, but is also useful for torch dataset.

    .. note::

        To include additional processing, please implement ``process`` method.

    """

    def __init__(
        self,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        self.decode_audio_as_waveform = decode_audio_as_waveform
        self.decode_audio_as_monoral = decode_audio_as_monoral

    def decode(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        from . import rename_webdataset_keys

        for key in sample.keys():
            # ported from
            # https://github.com/webdataset/webdataset/blob/f11fd66c163722c607ec99475a6f3cb880ec35b8/webdataset/autodecode.py#L418-L434
            ext = re.sub(r".*[.]", "", key)

            if ext in supported_audio_extensions:
                data = sample[key]

                if isinstance(data, bytes):
                    sample[key] = decode_audio(
                        data,
                        ext,
                        decode_audio_as_monoral=self.decode_audio_as_monoral,
                        decode_audio_as_waveform=self.decode_audio_as_waveform,
                    )

        sample = rename_webdataset_keys(sample)

        return sample

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process to edit each sample."""
        return sample

    def __call__(self, samples: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
        for sample in samples:
            sample = self.decode(sample)
            sample = self.process(sample)

            yield sample
