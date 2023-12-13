from ...cmudict import symbols as valid_cmudict_symbols

__all__ = [
    "full_symbols",
    "PAD_SYMBOL",
    "SPECIAL_SYMBOL",
    "PUNCTUATIONS",
    "ALPHABET_SYMBOLS",
    "ARPABET_SYMBOLS",
    "vocab_size",
]

PAD_SYMBOL = "_"
SPECIAL_SYMBOL = "-"
PUNCTUATIONS = [
    "!",
    "'",
    "(",
    ")",
    ",",
    ".",
    ":",
    ";",
    "?",
    " ",
]

# valid alphabet symbols
ALPHABET_SYMBOLS = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
]

# for ARPAbet symbols
# They are represented by "@SYMBOL".
ARPABET_SYMBOLS = ["@" + s for s in valid_cmudict_symbols]

full_symbols = [PAD_SYMBOL] + [SPECIAL_SYMBOL] + PUNCTUATIONS + ALPHABET_SYMBOLS + ARPABET_SYMBOLS

vocab_size = len(full_symbols)
