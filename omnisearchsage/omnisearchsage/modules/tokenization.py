from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set
from typing import Type

from enum import IntEnum
from functools import lru_cache

import pyarrow.parquet as pq
import torch
from torch import nn
from transformers import DistilBertTokenizer

if TYPE_CHECKING:
    from torch import Tensor
    from torch.jit import Final

TOKENIZER_CACHE_DIR = "/data1/huggingface_tokenizers/"


class BatchTokenizer(object):
    """
    Interface for text tokenization. The encode_batch function should return a str -> tensor mapping.
    In the case of a bert-style model, this might look like {text_ids: (N x S) LongTensor, mask: (N x S) BoolTensor}
    In the case of a bag of words model (using EmbeddingBag), this would probably look like
    {text_ids: (sum(lengths), ) LongTensor, offsets: cumsum(lengths) LongTensor}
    """

    def encode_batch(self, texts: List[str], **kwargs) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class TextNormalizationOption(IntEnum):
    LOWERCASE = 0
    UNICODE_NORMALIZE = 1
    TRIM_SPACE = 2
    COLLAPSE_WHITESPACE = 3


class BertTokenizerWrapper(nn.Module):
    def __init__(
        self,
        tokenizer_name: str,
        max_sequence_length: int,
        text_normalization_options: Set[TextNormalizationOption],
        additional_special_tokens: List[str] = (),
    ):
        super().__init__()
        bert_tokenizer = DistilBertTokenizer.from_pretrained(
            tokenizer_name, cache_dir=TOKENIZER_CACHE_DIR, use_fast=False
        )
        if additional_special_tokens:
            bert_tokenizer.add_special_tokens({"additional_special_tokens": additional_special_tokens})
        self._tokenizer_name = tokenizer_name
        self._max_sequence_length = max_sequence_length
        self._text_normalization_options = text_normalization_options
        self.tokenizer = torch.classes.pinterest_ops.BertTokenizer(
            bert_tokenizer.get_vocab(),
            max_sequence_length,
            [x.name for x in text_normalization_options],
            bert_tokenizer.basic_tokenizer.do_lower_case,
            bert_tokenizer.basic_tokenizer.tokenize_chinese_chars,
            bert_tokenizer.all_special_tokens,
        )

    def forward(self, texts: List[str], normalize: bool = True) -> Dict[str, Tensor]:
        return self.tokenizer.batch_encode(texts, normalize)

    def __len__(self) -> int:
        return len(self.tokenizer)

    def extra_repr(self) -> str:
        attributes = {
            "tokenizer_name": self._tokenizer_name,
            "vocab_size": len(self),
            "max_sequence_length": self._max_sequence_length,
            "text_normalization_options": ",".join(sorted([x.name for x in self._text_normalization_options])),
        }
        return ", ".join(f"{k}={v}" for k, v in attributes.items())


class EmbeddingBagTokenizer(BatchTokenizer, nn.Module):
    """
    Base tokenizer that assumes each input string is encoded as a list of int ids of variable length.
    The output is then encoded as some ids, and some offsets (i.e. the format expected by EmbeddingBag)
    """

    def encode_one(self, s: str) -> torch.Tensor:
        raise NotImplementedError

    @torch.jit.export
    def encode_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        offsets: List[int] = []
        mask: List[bool] = []

        cur_offset = 0
        id_list: List[torch.Tensor] = []
        for s in texts:
            ids = self.encode_one(s)
            offsets.append(cur_offset)
            mask.append(len(s) > 0)
            cur_offset += ids.size(0)
            id_list.append(ids)
        return {
            "input_ids": torch.cat(id_list, dim=0),
            "offsets": torch.tensor(offsets, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.bool),
        }

    def forward(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.encode_batch(texts=texts)


class VocabTokenizerWrapper(EmbeddingBagTokenizer):
    """
    Implementation of EmbeddingBagTokenizer that takes in a parquet file stored in `path` containing 2 columns:
    * "token", storing individual tokens
    * "idx" storing the index of each token

    if oov_size > 0, then oov tokens will be hashed and assigned a disjoint subset of ids from the vocab
    if max_vocab_size is set, then we trim the vocab to all tokens with idx <= max_vocab_size

    Classes that inherit from this should implement `tokenize_one`, which takes a single string, and returns
    a list of strings that are compatible with the underlying vocabulary of the tokenizer
    """

    def __init__(self, path: Optional[str], oov_size: int, max_vocab_size: Optional[int] = None):
        super().__init__()
        if path is None:
            assert max_vocab_size == 0
            assert oov_size > 0

        self.oov_size: Final[int] = oov_size
        self.max_vocab_size: Final[int] = max_vocab_size
        self._vocab_path: Final[str] = path

        if path is not None:
            pq_vocab_table = pq.read_table(path.rstrip("/")).to_pydict()
            vocab = dict(zip(pq_vocab_table["token"], pq_vocab_table["idx"]))

            assert len(vocab) == len(pq_vocab_table["token"]), (len(vocab), len(pq_vocab_table))
            assert set(vocab.values()) == set(range(len(vocab)))
        else:
            vocab = {}

        if max_vocab_size is not None:
            vocab = {k: v + 1 for k, v in vocab.items() if v < max_vocab_size}
        if len(vocab) > 0:
            assert min(vocab.values()) == 1, min(vocab.values())
        else:
            assert oov_size > 0
        self.vocab: Final[Dict[str, int]] = vocab
        self.tokenizer = torch.classes.pinterest_ops.VocabTokenizer({k: v - 1 for k, v in vocab.items()}, oov_size)

    @torch.jit.export
    def __len__(self) -> int:
        return len(self.tokenizer) + 1  # add 1 to be compatible with the non-fast version


class CharTrigramTokenizer(VocabTokenizerWrapper):
    def encode_one(self, s: str) -> torch.Tensor:
        return self.tokenizer.char_trigram_tokenize(s) + 1


class UnigramTokenizer(VocabTokenizerWrapper):
    def encode_one(self, s: str) -> torch.Tensor:
        return self.tokenizer.ngram_tokenize(s, 1) + 1


class BigramTokenizer(VocabTokenizerWrapper):
    def encode_one(self, s: str) -> torch.Tensor:
        return self.tokenizer.ngram_tokenize(s, 2) + 1


class TrigramTokenizer(VocabTokenizerWrapper):
    def encode_one(self, s: str) -> torch.Tensor:
        return self.tokenizer.ngram_tokenize(s, 3) + 1


class NgramTokenizerSpec(NamedTuple):
    path: str
    oov_size: int
    tok_cls: Type[VocabTokenizerWrapper]
    max_vocab_size: Optional[int] = None

    def create_tok(self):
        return self.tok_cls(path=self.path, oov_size=self.oov_size, max_vocab_size=self.max_vocab_size)


class MultiVocabTokenizer(EmbeddingBagTokenizer):
    """
    EmbeddingBagTokenizer that chains together several SimpleVocabTokenizers, assigning each to a
    disjoint range of output ids
    """

    TOKENIZER_VOCAB_PATH_BASE = "/omnisearchsage/omnisearchsage/modules/vocabs"

    @classmethod
    @lru_cache(4)
    def default(cls, source: str = "all") -> MultiVocabTokenizer:
        assert source in {"desc", "title", "query", "all"}
        return cls(
            [
                NgramTokenizerSpec(
                    path=f"{cls.TOKENIZER_VOCAB_PATH_BASE}/source={source}/token_type=char_trigram",
                    oov_size=0,
                    max_vocab_size=64_000,
                    tok_cls=CharTrigramTokenizer,
                ),
                NgramTokenizerSpec(
                    path=f"{cls.TOKENIZER_VOCAB_PATH_BASE}/source={source}/token_type=unigram",
                    oov_size=0,
                    max_vocab_size=200_000,
                    tok_cls=UnigramTokenizer,
                ),
                NgramTokenizerSpec(
                    path=f"{cls.TOKENIZER_VOCAB_PATH_BASE}/source={source}/token_type=bigram",
                    oov_size=0,
                    max_vocab_size=1_000_000,
                    tok_cls=BigramTokenizer,
                ),
            ]
        )

    def __init__(self, tokenizer_specs: List[NgramTokenizerSpec]) -> None:
        super().__init__()
        self._tokenizers = nn.ModuleList(
            [
                torch.jit.script(s.create_tok())
                for s in tokenizer_specs
                if s.max_vocab_size is None or (s.oov_size + s.max_vocab_size) > 0
            ]
        )
        assert self._tokenizers, tokenizer_specs
        self._total_size: Final[int] = sum(map(len, self._tokenizers))
        id_offsets: List[int] = [0]
        for tok in self._tokenizers:
            id_offsets.append(len(tok) + id_offsets[-1])
        self._id_offsets: Final[List[int]] = id_offsets

    def __len__(self) -> int:
        return self._total_size

    def encode_one(self, s: str) -> torch.Tensor:
        all_ids = []

        for ix, tok in enumerate(self._tokenizers):
            all_ids.append(tok.encode_one(s) + self._id_offsets[ix])
        return torch.cat(all_ids, dim=0)
