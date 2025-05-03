from typing import List, Union


class AudioSetIndexer:
    """Indexer to map tag into index.

    Args:
        tag_to_index (dict): Dictionary to map tag to index.

    Examples:

        >>> from audyn.utils.data.audioset import AudioSetIndexer, tag_to_index, name_to_index
        >>> indexer = AudioSetIndexer(tag_to_index)
        >>> indexer("/m/09x0r")
        0
        >>> indexer(["/m/09x0r", "/m/07hvw1"])
        [0, 526]
        >>> indexer = AudioSetIndexer(name_to_index)
        >>> indexer("Speech")
        0
        >>> indexer(["Speech", "Field recording"])
        [0, 526]

    """

    def __init__(self, name_to_index: dict[str, int]) -> None:
        super().__init__()

        index_to_name = {}

        for name, index in name_to_index.items():
            assert index not in index_to_name

            index_to_name[index] = name

        self.name_to_index = name_to_index
        self.index_to_name = index_to_name

    def __call__(self, name_or_names: Union[str, List[str]]) -> Union[int, List[int]]:
        index = self.encode(name_or_names)

        return index

    def __len__(self) -> int:
        return len(self.name_to_index)

    def encode(self, name_or_names: Union[str, List[str]]) -> Union[int, List[int]]:
        index = self._name_to_index(name_or_names)

        return index

    def decode(self, index_or_indices: Union[int, List[int]]) -> Union[str, List[str]]:
        tag = self._index_to_name(index_or_indices)

        return tag

    def _name_to_index(self, name_or_names: Union[str, List[str]]) -> Union[int, List[int]]:
        if isinstance(name_or_names, str):
            index = self.name_to_index[name_or_names]

            return index
        else:
            indices = []

            for name in name_or_names:
                index = self._name_to_index(name)
                indices.append(index)

            return indices

    def _index_to_name(self, index_or_indices: Union[int, list[int]]) -> Union[str, list[str]]:
        if isinstance(index_or_indices, int):
            tag = self.index_to_name[index_or_indices]

            return tag
        else:
            tags = []

            for index in index_or_indices:
                tag = self._index_to_name(index)
                tags.append(tag)

            return tags

    @classmethod
    def build_from_default_config(cls, type: str) -> "AudioSetIndexer":
        """Build AudioSetIndexer from default config.

        Args:
            type (str): Type of dataset. Only ``"default-tag"`` and ``"default-name"``
                are supported.

        Examples:

            >>> from audyn.utils.data.audioset import AudioSetIndexer
            >>> indexer = AudioSetIndexer.build_from_default_config("default-tag")
            >>> indexer("/m/09x0r")
            0
            >>> indexer(["/m/09x0r", "/m/07hvw1"])
            [0, 526]
            >>> indexer = AudioSetIndexer.build_from_default_config("default-name")
            >>> indexer("Speech")
            0
            >>> indexer(["Speech", "Field recording"])
            [0, 526]

        """
        from . import name_to_index as default_name_to_index
        from . import tag_to_index as default_tag_to_index

        if type == "default-tag":
            name_to_index = default_tag_to_index
        elif type == "default-name":
            name_to_index = default_name_to_index
        else:
            raise ValueError(f"{type} is not supported as type.")

        return cls(name_to_index)
