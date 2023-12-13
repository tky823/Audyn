import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from .numbers import normalize_numbers

__all__ = [
    "TacotronCleaner",
    "TacotronBasicCleaner",
    "TacotronTransliterationCleaner",
    "TacotronEnglishCleaner",
]

ABBREVIATIONS = {
    "mrs": "misess",
    "mr": "mister",
    "dr": "doctor",
    "st": "saint",
    "co": "company",
    "jr": "junior",
    "maj": "major",
    "gen": "general",
    "drs": "doctors",
    "rev": "reverend",
    "lt": "lieutenant",
    "hon": "honorable",
    "sgt": "sergeant",
    "capt": "captain",
    "esq": "esquire",
    "ltd": "limited",
    "col": "colonel",
    "ft": "fort",
}
_whitespace_re = re.compile(r"\s+")


class TacotronCleaner(ABC):
    def __init__(
        self,
        abbreviations: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()

        if abbreviations is None:
            abbreviations = ABBREVIATIONS

        self.abbreviations_re = [
            (re.compile("\\b{}\\.".format(short), re.IGNORECASE), long)
            for short, long in ABBREVIATIONS.items()
        ]

    def __call__(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        if isinstance(text, list):
            cleaned_text = []

            for _text in text:
                _cleaned_text = self._clean(_text)
                cleaned_text.append(_cleaned_text)
        else:
            cleaned_text = self._clean(text)

        return cleaned_text

    @abstractmethod
    def _clean(self, text: str) -> str:
        raise NotImplementedError("_clean is not implemented.")

    def expand_abbreviations(self, text: str) -> str:
        for regex, replacement in self.abbreviations_re:
            text = re.sub(regex, replacement, text)
        return text

    def expand_numbers(self, text: str) -> str:
        return normalize_numbers(text)

    def lowercase(self, text: str) -> str:
        return text.lower()

    def collapse_whitespace(self, text: str) -> str:
        return re.sub(_whitespace_re, " ", text)


class TacotronBasicCleaner(TacotronCleaner):
    def _clean(self, text: str) -> str:
        text = self.lowercase(text)
        cleaned_text = self.collapse_whitespace(text)

        return cleaned_text


class TacotronTransliterationCleaner(TacotronCleaner):
    def _forward(self, text: str) -> str:
        text = self.lowercase(text)
        cleaned_text = self.collapse_whitespace(text)

        return cleaned_text


class TacotronEnglishCleaner(TacotronCleaner):
    def _clean(self, text: str) -> str:
        text = self.lowercase(text)
        text = self.expand_numbers(text)
        text = self.expand_abbreviations(text)
        cleaned_text = self.collapse_whitespace(text)

        return cleaned_text


class TacotronBasicNormalizer(TacotronBasicCleaner):
    """alias of TacotronBasicCleaner."""


class TacotronTransliterationNormalizer(TacotronTransliterationCleaner):
    """alias of TacotronTransliterationCleaner."""


class TacotronEnglishNormalizer(TacotronEnglishCleaner):
    """alias of TacotronEnglishCleaner."""
