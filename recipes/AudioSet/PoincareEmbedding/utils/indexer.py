from audyn.utils.data.audioset import AudioSetIndexer as _AudioSetIndexer


class AudioSetIndexer(_AudioSetIndexer):
    @classmethod
    def build_from_default_config(cls, type: str) -> "AudioSetIndexer":
        """Build AudioSetIndexer from default config.

        Args:
            type (str): Type of dataset. Only ``"default"`` is supported.

        Examples:

            >>> from utils.indexer import AudioSetIndexer
            >>> indexer = AudioSetIndexer.build_from_default_config("default")
            >>> indexer("A capella")
            0
            >>> indexer(["A capella", "Zither"])
            [0, 632]

        """
        from . import name_to_index as default_name_to_index

        if type == "default":
            name_to_index = default_name_to_index
        else:
            raise ValueError(f"{type} is not supported as type.")

        return cls(name_to_index)
