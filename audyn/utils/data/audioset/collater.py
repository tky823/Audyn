from typing import Any, Dict, List

import torch

from .. import default_collate_fn, rename_webdataset_keys
from ..collater import BaseCollater
from . import num_tags as num_audioset_tags
from . import tags as audioset_tags

__all__ = ["AudioSetMultiLabelCollater"]


class AudioSetMultiLabelCollater(BaseCollater):
    """Collater to include multi-labels of AudioSet.

    Args:
        tags_key (str): Key of tags in given sample.
        multilabel_key (str): Key of multi-label to add to given sample.

    """

    def __init__(self, tags_key: str, multilabel_key: str) -> None:
        super().__init__()

        self.tags_key = tags_key
        self.multilabel_key = multilabel_key

        tag_to_index = {}

        for idx, tag in enumerate(audioset_tags):
            _tag = tag["tag"]
            tag_to_index[_tag] = idx

        self.tag_to_index = tag_to_index

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        tags_key = self.tags_key

        for sample in batch:
            sample = rename_webdataset_keys(sample)
            tags = sample[tags_key]
            labels = torch.zeros(
                (num_audioset_tags,),
                dtype=torch.long,
            )

            for tag in tags:
                tag_idx = self.tag_to_index[tag]
                labels[tag_idx] = 1

            sample[self.multilabel_key] = labels

        dict_batch = default_collate_fn(batch)

        return dict_batch
