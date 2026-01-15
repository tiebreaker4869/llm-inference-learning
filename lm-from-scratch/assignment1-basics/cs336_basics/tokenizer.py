from abc import ABC
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def pretokenize(string: str):
    for match in re.finditer(PAT, string):
        yield string[match.start():match.end()]

class Tokenizer(ABC):
    def encode(self, string: str) -> list[int]:
        """Encode a string to a list of tokens

        Args:
            string (str): input string

        Returns:
            list[int]: tokenized input
        """
        pass
    def decode(self, tokens: list[int]) -> str:
        """Decode a list of tokens to string

        Args:
            tokens (list[int]): token indices

        Returns:
            str: decoded string
        """
        pass