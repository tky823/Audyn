from audyn.utils.data.cmudict import BOS_IDX, BOS_SYMBOL, EOS_IDX, EOS_SYMBOL, CMUDict
from audyn.utils.data.cmudict.indexing import CMUDictIndexer


def test_cmudict() -> None:
    pron_dict = CMUDict()

    assert pron_dict["Hello"] == ["HH", "AH0", "L", "OW1"]


def test_cmudict_indexer() -> None:
    """Ensure order of full_symbols and CMUDictIndexer."""
    pron_dict = CMUDict()
    phonemes = pron_dict["Hello"]
    phonemes.insert(0, BOS_SYMBOL)
    phonemes.append(EOS_SYMBOL)
    indexer = CMUDictIndexer()
    indices = indexer(phonemes)

    assert indices[0] == BOS_IDX
    assert indices[-1] == EOS_IDX
    assert indices[1:-1] == [42, 9, 53, 59]
