from .tokenizer import Tokenizer, pretokenize
from dataclasses import dataclass
import regex as re
from typing import Iterable, Iterator
import json

@dataclass(frozen=True)
class BPETokenizerParams:
    """All params needed to specify a bpe tokenizer."""
    vocab: dict[int, bytes] # index -> bytes
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None
    
class BPETokenizer(Tokenizer):
    def __init__(self, params: BPETokenizerParams):
        self.vocab = params.vocab
        self.special_tokens = params.special_tokens
        self.special_token_to_idx = dict()
        self.special_tokens_regex = None
        if self.special_tokens:
            self._init_special_tokens()
        self.bytes_to_idx = {bs : idx for idx, bs in self.vocab.items()}
        self._init_merges(params.merges)
        
    def _init_merges(self, merges: list[tuple[bytes, bytes]]):
        self.merges_idx: list[tuple[tuple[int, int], int]] = []
        for b1, b2 in merges:
            idx1, idx2 = self.bytes_to_idx[b1], self.bytes_to_idx[b2]
            merge_idx = self.bytes_to_idx[b1 + b2]
            self.merges_idx.append(((idx1, idx2), merge_idx))

    def _init_special_tokens(self):
        # initialize special token regex
        escaped_tokens = [re.escape(st) for st in self.special_tokens]
        escaped_tokens.sort(key = len, reverse = True)
        self.special_tokens_regex = "(" + "|".join(escaped_tokens) + ")"
        
        # add new special tokens
        special_tokens_set = set(self.special_tokens)
        bytes_special_tokens_set = set({s.encode("utf-8") for s in special_tokens_set})
        in_vocab_special_tokens = set({t for t in self.vocab.values()if t in bytes_special_tokens_set})
        in_vocab_special_tokens = set({bs.decode("utf-8") for bs in in_vocab_special_tokens})
        nxt_idx = max(self.vocab.keys()) + 1
        new_special_tokens = special_tokens_set - in_vocab_special_tokens
        for new_special_token in new_special_tokens:
            self.vocab[nxt_idx] = new_special_token.encode("utf-8")
            nxt_idx += 1
        
        # add reverse mapping for special tokens
        for token, bstring in self.vocab.items():
            if bstring in bytes_special_tokens_set:
                string = bstring.decode("utf-8")
                self.special_token_to_idx[string] = token

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        vocab = dict()
        merges = []
        with open(vocab_filepath, mode="r") as f:
            content = f.read()
            vocab = json.loads(content)
            vocab = {v: k.encode("utf-8") for k, v in vocab.items()}
        with open(merges_filepath, mode="r") as f:
            for line in f:
                cleaned_line = line.strip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    merges.append(tuple([s.encode("utf-8") for s in cleaned_line.split(" ")]))
        params = BPETokenizerParams(vocab, merges, special_tokens)
        tokenizer = BPETokenizer(params)
        return tokenizer
        
    def encode(self, string: str) -> list[int]:
        splitted = re.split(self.special_tokens_regex, string) if self.special_tokens else [string]
        indices = []
        for chunk in splitted:
            if not chunk:
                continue
            if chunk in self.special_token_to_idx:
                indices.append(self.special_token_to_idx[chunk])
            else:
                pretokens = pretokenize(chunk)
                for pretoken in pretokens:
                    if not pretoken:
                        continue
                    bs = pretoken.encode("utf-8")
                    bs_list = [bs[i:i+1] for i in range(len(bs))]
                    idxs = [self.bytes_to_idx[b] for b in bs_list]
                    for pair, new_idx in self.merges_idx:
                        idxs = self._merge(idxs, pair, new_idx)
                    indices.extend(idxs)
        return indices
    def decode(self, tokens: list[int]) -> str:
        bytes_list = [self.vocab[token] for token in tokens]
        string = b"".join(bytes_list).decode("utf-8", errors='replace')
        return string
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for string in iterable:
            tokens = self.encode(string)
            for token in tokens:
                yield token
    
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