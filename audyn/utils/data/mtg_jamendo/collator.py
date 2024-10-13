from typing import Any, Dict, List, Optional, Union

from audyn.utils.data.composer import Composer

from ..collator import Collator


class MTGJamendoEvaluationCollator(Collator):
    """Collater for evaluation of MTG-Jamendo.

    Args:
        composer (Composer, optional): Composer that processes samples.
        squeezed_key (str or list, optional): List of keys to be squeezed.

    """

    def __init__(
        self,
        composer: Optional[Composer] = None,
        squeezed_key: Optional[Union[str, List[str]]] = None,
    ) -> None:
        super().__init__(composer=composer)

        if squeezed_key is None:
            squeezed_keys = []
        elif isinstance(squeezed_key, str):
            squeezed_keys = [squeezed_key]
        else:
            squeezed_keys = squeezed_key

        self.squeezed_keys = squeezed_keys

    def __call__(self, batch: List[Any]) -> Dict[str, Any]:
        keys = self.squeezed_keys

        dict_batch = super().__call__(batch)

        for key in keys:
            value = dict_batch[key]

            assert len(value) == 1

            dict_batch[key] = value[0]

        return dict_batch
