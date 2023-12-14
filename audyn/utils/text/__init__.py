from typing import List, Optional, Union

import torch
import torch.nn as nn

__all__ = ["load_text", "TextPreprocessor"]


def load_text(path: str, split: Optional[str] = None) -> List[str]:
    """Load text file.

    Args:
        path (str): Path to text file.

    Returns:
        list: List of words written in text file.

    .. code-block:: shell

        echo "All in the golden afternoon" > "sample.txt"
        python
        >>> from audyn.utils.text import load_text
        >>> load_text("sample.txt")
        'All in the golden afternoon'
        >>> load_text("sample.txt", split=" ")
        ['All', 'in', 'the', 'golden', 'afternoon']

    """
    with open(path) as f:
        lines = f.readlines()

    if len(lines) != 1:
        raise ValueError(f"The text format of {path} is incorrect.")

    line = lines[0]
    text = line.strip()

    if split is not None:
        text = text.split(split)

    return text


class TextPreprocessor(nn.Module):
    """TextPreprocessor to normalize, tokenize, phonemize, and index text.

    Args:
        normalizer (nn.Module): Text normalizer that cleans text.
        tokenizer (nn.Module): Tokenizer that splits text (sentence) into list of tokens.
        phonemizer (nn.Module): Module to convert grapheme sequence to phoneme one.
        indexer (nn.Module): Module to map phoneme sequence to numeric sequence.

    .. note::

        In this module, tokenizer does not output numeric values, but list of strings.

    """

    def __init__(
        self,
        normalizer: Optional[nn.Module] = None,
        tokenizer: Optional[nn.Module] = None,
        phonemizer: Optional[nn.Module] = None,
        indexer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.normalizer = normalizer
        self.tokenizer = tokenizer
        self.phonemizer = phonemizer
        self.indexer = indexer

    def forward(
        self, text: str, return_type: Union[str, type] = "tensor"
    ) -> Union[List[str], torch.Tensor]:
        """Transform text to index sequence.

        Args:
            text (str): Text sequence.

        Returns:
            list or torch.Tensor: Transformed index sequence.

        """
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_normalized_text(normalized_text)
        phonemes = self.phonemize_tokens(tokens)
        indices = self.index_phonemes(phonemes, return_type=return_type)

        return indices

    def normalize_text(self, text: str) -> str:
        """Normalize text by ``normalizer`` to expand abbreviations, notation of numbers, etc.

        Args:
            text (str): Text to be normalized.

        Returns:
            str: Normalized text.

        """
        if self.normalizer is None:
            normalized_text = text
        else:
            normalized_text = self.normalizer(text)

        return normalized_text

    def tokenize_normalized_text(self, normalized_text: str) -> List[str]:
        """Tokenize normalized text by ``tokenizer``.

        Args:
            text (str): Text to be tokenized.

        Returns:
            list: Tokenized text.

        """
        if self.tokenizer is None:
            tokens = normalized_text
        else:
            tokens = self.tokenizer(normalized_text)

        return tokens

    def phonemize_tokens(self, tokens: List[str]) -> List[str]:
        """Convert grapheme sequence (tokens) to phoneme one.

        Args:
            tokens (list): Tokens to be converted to phoneme sequence.

        Returns:
            list: Phoneme sequence.

        """
        if self.phonemizer is None:
            phonemes = tokens
        else:
            phonemes = self.phonemizer(tokens)

        return phonemes

    def index_phonemes(
        self,
        phonemes: List[str],
        return_type: Union[str, type] = int,
    ) -> Union[List[int], torch.Tensor]:
        """Map phone in phoneme sequence to numerical values.

        Args:
            phonemes (list): Phoneme sequence.
            return_type (str or type): Return type.

        Returns:
            list or torch.Tensor: Numerical sequence representing phoneme sequence.

        """
        if self.indexer is None:
            indices = phonemes
        else:
            indices = self.indexer(phonemes)

        if return_type == int or return_type == "int":
            pass
        elif return_type == "tensor" or return_type in [torch.Tensor, torch.LongTensor]:
            indices = self.to_tensor(indices)
        else:
            raise NotImplementedError(f"return_type {return_type} is not supported.")

        return indices

    def tokenize_text(self, text: str) -> List[str]:
        """Sequential operation of ``normalize_text`` and ``tokenize_normalized_text``.

        Args:
            text (str): Text to be normalized and tokenized.

        Returns:
            list: Tokenized text.

        """
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_normalized_text(normalized_text)

        return tokens

    def phonemize_normalized_text(self, normalized_text: str) -> List[str]:
        """Sequential operation of ``tokenize_normalized_text`` and ``phonemize_tokens``.

        Args:
            text (str): Normalized text to be converted to phoneme sequence.

        Returns:
            list: Phoneme sequence.

        """
        tokens = self.tokenize_normalized_text(normalized_text)
        phonemes = self.phonemize_tokens(tokens)

        return phonemes

    def index_tokens(
        self,
        tokens: List[str],
        return_type: Union[str, type] = int,
    ) -> Union[List[int], torch.Tensor]:
        """Sequential operation of ``phonemize_tokens`` and ``index_phonemes``.

        Args:
            text (str): Tokens to be converted to phoneme sequence
                and then mapped to numerical sequence.

        Returns:
            list or torch.Tensor: Numerical sequence representing phoneme sequence.

        """
        phonemes = self.phonemize_tokens(tokens)
        indices = self.index_phonemes(phonemes, return_type=return_type)

        return indices

    def phonemize_text(self, text: str) -> List[str]:
        """Sequential operation of ``normalize_text``,
        ``tokenize_normalized_text`` and ``phonemize_tokens``.

        Args:
            text (str): Text to be converted to phoneme sequence.

        Returns:
            list: Phoneme sequence.

        """
        normalized_text = self.normalize_text(text)
        tokens = self.tokenize_normalized_text(normalized_text)
        phonemes = self.phonemize_tokens(tokens)

        return phonemes

    def index_normalized_text(
        self,
        normalized_text: str,
        return_type: Union[str, type] = int,
    ) -> Union[List[int], torch.Tensor]:
        """Sequential operation of ``tokenize_normalized_text``,
        ``phonemize_tokens`` and ``index_phonemes``.

        Args:
            text (str): Normalized text.

        Returns:
            list or torch.Tensor: Numerical sequence representing phoneme sequence.

        """
        tokens = self.tokenize_normalized_text(normalized_text)
        phonemes = self.phonemize_tokens(tokens)
        indices = self.index_phonemes(phonemes, return_type=return_type)

        return indices

    def to_tensor(self, input: Union[int, List[int]]) -> torch.LongTensor:
        """Transform int or list input to torch.LongTensor.

        Args:
            input (int or list): Input int value or sequence.

        Returns:
            torch.LongTensor: Transformed long tensor.

        """
        output = torch.tensor(input, dtype=torch.long)

        return output
