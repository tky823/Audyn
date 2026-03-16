from typing import Any, Dict

from ....transforms.clip import OpenAICLIPImageTransform
from ..composer import Composer

__all__ = [
    "OpenAICLIPImageEncoderComposer",
    "OpenAIImageEncoderComposer",
]


class OpenAICLIPImageEncoderComposer(Composer):
    """Composer for OpenAICLIPImageEncoder.

    Args:
        input_key (str): Key of image in given sample.
        output_key (str): Key of image to store sample.

    """

    def __init__(
        self,
        input_key: str,
        output_key: str,
        size: int = 224,
        *,
        training: bool = False,
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.transform = OpenAICLIPImageTransform(size=size)

        self.input_key = input_key
        self.output_key = output_key

        if training:
            self.transform.train()
        else:
            self.transform.eval()

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key

        image = sample[input_key]
        sample[output_key] = self.transform(image)

        return sample


class OpenAIImageEncoderComposer(OpenAICLIPImageEncoderComposer):
    """Alias of OpenAICLIPImageEncoderComposer."""
