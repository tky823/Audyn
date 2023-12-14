from typing import Dict, List, Union

__all__ = ["Phonemizer"]


class Phonemizer:
    """Grapheme-to-phoneme (a.k.a. G2P) converter.

    Args:
        pron_dict (dict): Pronunciation dictionary that takes a word and returns
            pronunciation (str) or candidates of pronunciations (list of str).
        unk_token (str): Token to represent unkwon words.

    """

    def __init__(
        self,
        pron_dict: Dict[str, Union[str, List[str]]],
        unk_token: str = "<UNK>",
    ) -> None:
        self.pron_dict = pron_dict

        self.unk_token = unk_token

    def __call__(self, text: List[str]) -> List[str]:
        """Text normalization and tokenization should be applied
        before performing G2P."""
        pron_dict = self.pron_dict
        unk_token = self.unk_token

        is_dict = isinstance(pron_dict, dict)

        phones = []

        for token in text:
            if is_dict:
                phone = pron_dict.get(token, unk_token)
            else:
                phone = pron_dict[token]

                if phone is None:
                    phone = unk_token

            if isinstance(phone, str):
                phones.append(phone)
            elif isinstance(phone, list):
                phones = phones + phone

        return phones
