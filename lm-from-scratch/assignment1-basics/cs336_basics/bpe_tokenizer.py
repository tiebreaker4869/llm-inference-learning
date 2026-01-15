from .tokenizer import Tokenizer, pretokenize
from dataclasses import dataclass

@dataclass(frozen=True)
class BPETokenizerParams:
    """All params needed to specify a bpe tokenizer."""
    vocab: dict[int, bytes] # index -> bytes
    merges: dict[tuple[int, int], int] # index1, index2 -> new_index
    
class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params
    def encode(self, string: str) -> list[int]:
        pretokens = pretokenize(string)
        indices = []
        for pretoken in pretokens:
            idxs = list(map(int, pretoken.encode("utf-8")))
            for pair, new_idx in self.params.merges.items():
                idxs = self._merge(idxs, pair, new_idx)
            indices.extend(idxs)
        return indices
    def decode(self, tokens: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, tokens))
        string = b"".join(bytes_list).decode("utf-8")
        return string
    
    @staticmethod
    def _merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
        merged = []
        i = 0
        while i < len(indices):
            if i + 1 < len(indices) and (indices[i], indices[i+1]) == pair:
                merged.append(new_index)
                i += 2
            else:
                merged.append(indices[i])
                i += 1
        return merged