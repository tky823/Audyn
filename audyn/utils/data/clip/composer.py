from typing import Any, Dict

import torchvision.transforms as vT

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
        decode_audio_as_waveform: bool = True,
        decode_audio_as_monoral: bool = True,
    ) -> None:
        super().__init__(
            decode_audio_as_waveform=decode_audio_as_waveform,
            decode_audio_as_monoral=decode_audio_as_monoral,
        )

        self.transforms = vT.Compose(
            [
                vT.Resize(size, interpolation=vT.InterpolationMode.BICUBIC),
                vT.CenterCrop(size),
                vT.ToTensor(),
                vT.Normalize(
                    [0.48145466, 0.4578275, 0.40821073],
                    [0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )

        self.input_key = input_key
        self.output_key = output_key

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        input_key = self.input_key
        output_key = self.output_key

        image = sample[input_key]
        sample[output_key] = self.transforms(image)

        return sample


class OpenAIImageEncoderComposer(OpenAICLIPImageEncoderComposer):
    """Alias of OpenAICLIPImageEncoderComposer."""
