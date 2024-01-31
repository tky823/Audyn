from audyn.utils.data.clotho import ClothoTextPreprocessor
from audyn.utils.data.clotho.text.indexing import ClothoTextIndexer
from audyn.utils.data.clotho.text.normalization import ClothoTextNormalizer
from audyn.utils.data.clotho.text.symbols import BOS_SYMBOL, EOS_SYMBOL
from audyn.utils.data.clotho.text.tokenization import ClothoTextTokenizer


def test_clotho_text_preprocessor() -> None:
    text_preprocessor = ClothoTextPreprocessor()

    text = "A car passes in front of me, then a bird starts singing."
    expected_indices = [0, 2, 525, 2553, 1857, 1535, 2432, 2206, 3846, 2, 314, 3612, 3352, 1]

    indices = text_preprocessor(text, return_type=int)

    assert indices == expected_indices


def test_clotho_text_normalizer() -> None:
    normalizer = ClothoTextNormalizer()

    raw_text = "Hello, world."
    expected_text = "hello world"

    normalized_text = normalizer(raw_text)

    assert normalized_text == expected_text


def test_clotho_text_tokenizer() -> None:
    tokenizer = ClothoTextTokenizer()

    normalized_text = "hello world"
    expected_tokens = ["hello", "world"]

    tokens = tokenizer(normalized_text)

    assert tokens == expected_tokens


def test_clotho_text_indexer() -> None:
    indexer = ClothoTextIndexer()

    tokens = [
        "a",
        "car",
        "passes",
        "in",
        "front",
        "of",
        "me",
        "then",
        "a",
        "bird",
        "starts",
        "singing",
    ]
    inserted_tokens = [BOS_SYMBOL] + tokens + [EOS_SYMBOL]
    expected_indices = [2, 525, 2553, 1857, 1535, 2432, 2206, 3846, 2, 314, 3612, 3352]
    expected_inserted_indices = [0] + expected_indices + [1]

    indices = indexer(tokens, insert_bos_token=False, insert_eos_token=False)
    assert indices == expected_indices

    indices = indexer(tokens, insert_bos_token=True, insert_eos_token=True)
    assert indices == expected_inserted_indices

    indices = indexer(inserted_tokens, insert_bos_token=False, insert_eos_token=False)
    assert indices == expected_inserted_indices

    indices = indexer(inserted_tokens, insert_bos_token=True, insert_eos_token=True)
    assert indices == expected_inserted_indices
