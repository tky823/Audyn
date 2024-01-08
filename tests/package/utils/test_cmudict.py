from audyn.utils.data.cmudict import CMUDict
from audyn.utils.data.cmudict.indexing import CMUDictIndexer


def test_cmudict() -> None:
    pron_dict = CMUDict()

    assert pron_dict["Hello"] == ["HH", "AH0", "L", "OW1"]


def test_cmudict_indexer() -> None:
    pron_dict = CMUDict()
    phonemes = pron_dict["Hello"]
    indexer = CMUDictIndexer()
    indices = indexer(phonemes)

    assert indices == [43, 10, 54, 60]
