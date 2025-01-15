from audyn.utils.data.tacotron import (
    TacotronEnglishCleaner,
    TacotronIndexer,
    TacotronTextTokenizer,
)


def test_tacotron() -> None:
    normalizer = TacotronEnglishCleaner()
    tokenizer = TacotronTextTokenizer()
    indexer = TacotronIndexer()

    text = "Hello, 2025."
    normalized_text = normalizer(text)
    tokens = tokenizer(normalized_text)
    indices = indexer(tokens)

    assert normalized_text == "hello, twenty twenty-five."
    assert tokens == [
        "h",
        "e",
        "l",
        "l",
        "o",
        ",",
        " ",
        "t",
        "w",
        "e",
        "n",
        "t",
        "y",
        " ",
        "t",
        "w",
        "e",
        "n",
        "t",
        "y",
        "-",
        "f",
        "i",
        "v",
        "e",
        ".",
    ]
    assert indices == [
        45,
        42,
        49,
        49,
        52,
        6,
        11,
        57,
        60,
        42,
        51,
        57,
        62,
        11,
        57,
        60,
        42,
        51,
        57,
        62,
        1,
        43,
        46,
        59,
        42,
        7,
    ]
