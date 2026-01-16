from .tokenizer import Tokenizer, pretokenize
from dataclasses import dataclass
import regex as re

@dataclass(frozen=True)
class BPETokenizerParams:
    """All params needed to specify a bpe tokenizer."""
    vocab: dict[int, bytes] # index -> bytes
    merges: list[tuple[int, int], int] # index1, index2 -> new_index
    special_tokens: list[str] | None
    
class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.params = params
        self.bytes_to_token = {bs : idx for idx, bs in self.params.vocab.items()}
        self.special_tokens_regex = re.escape("|").join(self.params.special_tokens) if self.params.special_tokens else ""
        self.special_token_to_idx = dict()
        if self.params.special_tokens:
            self._init_special_tokens()

    def _init_special_tokens(self):
        special_tokens_set = set(self.params.special_tokens)
        for token, bstring in self.params.vocab.items():
            string = bstring.decode("utf-8")
            if string in special_tokens_set:
                self.special_token_to_idx[string] = token
        
    def encode(self, string: str) -> list[int]:
        pretokens = pretokenize(string)
        indices = []
        for pretoken in pretokens:
            parts = re.split(self.special_tokens_regex, pretoken) if self.params.special_tokens else [pretoken]
            for part in parts:
                if part in self.special_token_to_idx:
                    indices.append(self.special_token_to_idx[part])
                else:
                    bs = part.encode("utf-8")
                    bs_list = [bs[i:i+1] for i in range(len(bs))]
                    idxs = [self.bytes_to_token[b] for b in bs_list]
                    for pair, new_idx in self.params.merges:
                        idxs = self._merge(idxs, pair, new_idx)
                    indices.extend(idxs)
        return indices
    def decode(self, tokens: list[int]) -> str:
        bytes_list = [self.params.vocab.get(token) for token in tokens]
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