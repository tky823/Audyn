import os
import re
import urllib
from typing import Dict, List, Optional

from ....utils import audyn_cache_dir

__all__ = [
    "symbols",
    "full_symbols",
    "BREAK_SYMBOLS",
    "BOS_SYMBOL",
    "EOS_SYMBOL",
    "UNK_SYMBOL",
    "PAD_SYMBOL",
    "BOS_IDX",
    "EOS_IDX",
    "UNK_IDX",
    "PAD_IDX",
    "SPECIALS",
    "vocab_size",
    "CMUDict",
]


symbols = [
    "AA",
    "AA0",
    "AA1",
    "AA2",
    "AE",
    "AE0",
    "AE1",
    "AE2",
    "AH",
    "AH0",
    "AH1",
    "AH2",
    "AO",
    "AO0",
    "AO1",
    "AO2",
    "AW",
    "AW0",
    "AW1",
    "AW2",
    "AY",
    "AY0",
    "AY1",
    "AY2",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "EH0",
    "EH1",
    "EH2",
    "ER",
    "ER0",
    "ER1",
    "ER2",
    "EY",
    "EY0",
    "EY1",
    "EY2",
    "F",
    "G",
    "HH",
    "IH",
    "IH0",
    "IH1",
    "IH2",
    "IY",
    "IY0",
    "IY1",
    "IY2",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OW0",
    "OW1",
    "OW2",
    "OY",
    "OY0",
    "OY1",
    "OY2",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UH0",
    "UH1",
    "UH2",
    "UW",
    "UW0",
    "UW1",
    "UW2",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
]
BREAK_SYMBOLS = ["sil", "spn"]
BOS_SYMBOL = "<BOS>"
EOS_SYMBOL = "<EOS>"
UNK_SYMBOL = "<UNK>"
PAD_SYMBOL = "<PAD>"
SPECIALS = [UNK_SYMBOL, PAD_SYMBOL]
full_symbols = symbols + BREAK_SYMBOLS + [BOS_SYMBOL] + [EOS_SYMBOL] + SPECIALS
BOS_IDX = full_symbols.index(BOS_SYMBOL)
EOS_IDX = full_symbols.index(EOS_SYMBOL)
UNK_IDX = full_symbols.index(UNK_SYMBOL)
PAD_IDX = full_symbols.index(PAD_SYMBOL)

vocab_size = len(full_symbols)


class CMUDict:
    """Pronunciation dictionary using CMUDict.

    For the detail about CMUDict,
    see http://www.speech.cs.cmu.edu/cgi-bin/cmudict.
    """

    filename = "cmudict-0.7b"
    cmudict_urls = ["https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b"]

    def __init__(self, root: Optional[str] = None) -> None:
        if root is None:
            root = os.path.join(audyn_cache_dir, "data", "cmudict")

        path = os.path.join(root, self.filename)

        if not os.path.exists(path):
            # TODO: additional sources
            assert len(self.cmudict_urls) == 1

            os.makedirs(root, exist_ok=True)

            for url in self.cmudict_urls:
                data = urllib.request.urlopen(url).read()

                with open(path, mode="wb") as f:
                    f.write(data)

                break

        self.cmudict = self.build_dict(path)

    def __getitem__(self, word: str) -> Optional[List[str]]:
        """Return phone from word.

        Args:
            word (str): Single word.

        Returns:
            list: Pronunciation. If pronunciation is unknown,
                dictionary returns ``None``.

        Examples:

            >>> from audyn.utils.data.cmudict import CMUDict
            >>> cmudict = CMUDict()
            >>> cmudict["Hello"]
            ['HH', 'AH0', 'L', 'OW1']

        """
        pronunciation_candidates = self.cmudict.get(word.upper())

        if pronunciation_candidates is None:
            phones = None
        else:
            phones = pronunciation_candidates[0]
            phones = phones.split(" ")

        return phones

    @staticmethod
    def build_dict(path: str) -> Dict[str, List[str]]:
        """Build pronunciation dictionary.

        Args:
            path (str): Path to dictionary to load.
        """
        cmudict = {}

        with open(path, encoding="latin-1") as f:
            for line in f:
                line = line.strip()

                if len(line) == 0:
                    continue

                if (line[0].isalpha() and line[0].isupper()) or line[0] == "'":
                    valid_word = True
                    word, pronunciation = line.split("  ")
                    word = re.sub(r"\(\d+\)", "", word)
                    phones = pronunciation.split(" ")

                    for p in phones:
                        if p not in symbols:
                            valid_word = False

                    if not valid_word:
                        continue

                    pronunciation = " ".join(phones)

                    if word not in cmudict:
                        cmudict[word] = []

                    cmudict[word].append(pronunciation)

        return cmudict
