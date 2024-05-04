import glob
import json
import os
import tarfile
import warnings
from typing import Dict, List, Optional

from torch.utils.data import WeightedRandomSampler

from . import tags as audioset_tags

__all__ = [
    "AudioSetWebDatasetWeightedRandomSampler",
]


class AudioSetWebDatasetWeightedRandomSampler(WeightedRandomSampler):
    """Weighted random sampler for AudioSet using WebDataset.

    The implementation is based on one described in [#koutini2022efficient]_.
    In this sampler, samples with rare tags are more likely to be taken.

    Args:
        feature_dir (str): Path to directory containing .tar files.
        num_samples (int): Number of samples at each epoch.
        replacement (bool): If ``True``, samples are taken with replacement.
        smooth (int): Offset to frequency of each class. In [#koutini2022efficient]_, ``1000``
            is used. Default: ``1``.
        ytids (list, optional): YouTube IDs. This list is useful to align order of samples between
            sampler and other modules. If ``None``, order of ytids are determined by
            alphabetical order using built-in ``sorted`` function.

    .. [#koutini2022efficient]
        K. Koutini et al., "Efficient training of audio transformers with patchout,"
        in *Interspeech*, 2022.

    """

    def __init__(
        self,
        feature_dir: str,
        num_samples: int,
        replacement: bool = True,
        smooth: float = 1,
        ytids: Optional[List[str]] = None,
        generator=None,
    ) -> None:
        weights_per_sample = _get_sampling_weights(feature_dir, smooth=smooth)

        if ytids is None:
            warnings.warn(
                "It is highly recommended to set ytids to align orders between "
                "sampler and other modules.",
                UserWarning,
                stacklevel=2,
            )
            ytids = sorted(list(weights_per_sample.keys()))

        # from dict to list
        weights = []

        for ytid in ytids:
            weight = weights_per_sample[ytid]
            weights.append(weight)

        super().__init__(
            weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )


def _get_sampling_weights(feature_dir: str, smooth: float) -> Dict[str, float]:
    tags_per_sample = {}
    frequency_per_tag = {}
    weight_per_sample = {}

    for tag in audioset_tags:
        _tag = tag["tag"]
        frequency_per_tag[_tag] = smooth

    for tar_path in sorted(glob.glob(os.path.join(feature_dir, "*.tar"))):
        with tarfile.open(tar_path) as f:
            for tarinfo in f:
                ytid, key = tarinfo.name.split(".", maxsplit=1)

                if key == "tags.json":
                    tags = f.extractfile(tarinfo).read()
                    tags = tags.decode("utf-8")
                    tags_per_sample[ytid] = json.loads(tags)
                    weight_per_sample[ytid] = 0

    for tags in tags_per_sample.values():
        for tag in tags:
            frequency_per_tag[tag] += 1

    for ytid, tags in tags_per_sample.items():
        for tag in tags:
            weight_per_sample[ytid] += 1 / frequency_per_tag[tag]

    return weight_per_sample
