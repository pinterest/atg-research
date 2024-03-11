from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple

import random
import unicodedata

import torch
from omnisearchsage.common.types import EntityType
from omnisearchsage.data.sage_features import SageBatch
from omnisearchsage.data.sage_features import TaskName
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from torch import Tensor


class RandomDataset(Iterator[SageBatch], IterableDataset):
    def __init__(
        self,
        string_features: Dict[EntityType, List[str]],
        num_examples: Optional[int] = None,
        batch_size: int = 1024,
        negative_ratio: int = 4,
        query_vocab_size: int = 256_000,
        vocab_size: int = 10_000_000,
        candidate_entity_type: EntityType = EntityType.SIGNATURE,
    ) -> None:
        self.string_features = string_features
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.vocab_size = vocab_size
        self.query_vocab_size = query_vocab_size
        self.unicode_glyphs = ''.join(
            chr(char)
            for char in range(65536)
            # use the unicode categories that don't include control codes
            if unicodedata.category(chr(char))[0] in 'LMNPSZ'
        )
        self._candidate_entity_type = candidate_entity_type
        self._batch = self._create_batch()
        self._num_examples = num_examples
        self._idx = 0

    def __next__(self) -> SageBatch:
        if self._num_examples is not None and (self._idx * self.batch_size) >= self._num_examples:
            raise StopIteration
        self._idx += 1
        return self._batch

    def _get_normalized_emb(self, *sizes: int) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.randn(*sizes, 256)).to(dtype=torch.float16)

    def _get_binary_emb(self, *sizes: int) -> torch.Tensor:
        return torch.randint(0, 255, (*sizes, 128), dtype=torch.uint8)

    def _get_multiple_embs(self, size: int, name: str, emb_fn: Callable) -> Dict[str, Tensor]:
        embs = emb_fn(size, 15)
        lengths = torch.randint(0, 15, (size,))
        mask = torch.arange(15).expand(size, 15) < lengths.unsqueeze(1)
        return {
            f"{name}_feat": embs,
            f"{name}_mask": mask.unsqueeze(2),
        }

    def _get_sparse_ids(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lengths = torch.randint(0, 500, (size,))
        ids = torch.randint(0, self.vocab_size, (lengths.sum(),), dtype=torch.int64)
        offsets = torch.cat([torch.tensor([0]), torch.cumsum(lengths, dim=0)[:-1]]).to(torch.int64)
        return ids, offsets

    def _get_tokenized_query(self, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized_ids = torch.randint(0, self.query_vocab_size, (size, 128), dtype=torch.int64)
        lengths = torch.randint(5, 128, (size,))
        mask = torch.arange(128).expand(size, 128) < lengths.unsqueeze(1)
        return tokenized_ids, mask

    def _get_text_features(self, size: int, feature_names: List[str]) -> Dict[str, Tensor]:
        output = {}
        for feature_name in feature_names:
            ids, offsets = self._get_sparse_ids(size)
            output[f"{feature_name}_input_ids"] = ids
            output[f"{feature_name}_offsets"] = offsets
        return output

    def _get_signatures(self, size: int) -> List[str]:
        # Sample 24 length random strings consisting of all printable ascii characters
        return ["".join([chr(i) for i in torch.randint(32, 127, (24,))]) for _ in range(size)]

    def _get_item_ids(self, size: int) -> List[str]:
        # Sample 48 length random strings consisting of all printable ascii characters
        return ["".join([chr(i) for i in torch.randint(32, 127, (48,))]) for _ in range(size)]

    def _unicode_fuzz(self, length: int) -> str:
        # generate it
        utf_string = ''.join([random.choice(self.unicode_glyphs) for _ in range(length)])

        return utf_string

    def _get_queries(self, size: int) -> List[str]:
        # sample lengths between 10 to 100
        lengths = torch.randint(10, 100, (size,))
        # sample random unicode strings according to the lengths
        return [self._unicode_fuzz(length) for length in lengths]

    def _get_query_signature_batch(self) -> SageBatch:
        candidate_batch_size = (1 + self.negative_ratio) * self.batch_size
        query_batch_size = self.batch_size

        signature_ids = self._get_signatures(candidate_batch_size)
        queries = self._get_queries(query_batch_size)
        tokenized_ids, mask = self._get_tokenized_query(query_batch_size)
        return SageBatch(
            keys={
                EntityType.SIGNATURE: signature_ids,
                EntityType.SEARCH_QUERY: queries,
            },
            tensor_feats={
                EntityType.SIGNATURE: {
                    "gs_v5": self._get_normalized_emb(candidate_batch_size),
                    "ue_v4": self._get_binary_emb(candidate_batch_size),
                    "item_is_v2": self._get_normalized_emb(candidate_batch_size),
                    "id_hash": torch.tensor([hash(s) for s in signature_ids], dtype=torch.int64),
                    **self._get_text_features(candidate_batch_size, self.string_features[EntityType.SIGNATURE]),
                },
                EntityType.SEARCH_QUERY: {
                    "query_text_input_ids": tokenized_ids,
                    "query_text_attention_mask": mask,
                    "id_hash": torch.tensor([hash(s) for s in queries], dtype=torch.int64),
                },
            },
            task_name=TaskName.METRIC_LEARNING,
            candidate_entity_type=EntityType.SIGNATURE,
            query_entity_type=EntityType.SEARCH_QUERY,
            num_pairs=query_batch_size,
        )

    def _get_query_query_batch(self) -> SageBatch:
        candidate_batch_size = (1 + self.negative_ratio) * self.batch_size
        query_batch_size = self.batch_size
        total_batch_size = candidate_batch_size + query_batch_size
        queries = self._get_queries(total_batch_size)
        tokenized_ids, mask = self._get_tokenized_query(total_batch_size)
        return SageBatch(
            keys={
                EntityType.SEARCH_QUERY: queries,
            },
            tensor_feats={
                EntityType.SEARCH_QUERY: {
                    "query_text_input_ids": tokenized_ids,
                    "query_text_attention_mask": mask,
                    "id_hash": torch.tensor([hash(s) for s in queries], dtype=torch.int64),
                },
            },
            task_name=TaskName.METRIC_LEARNING,
            candidate_entity_type=EntityType.SEARCH_QUERY,
            query_entity_type=EntityType.SEARCH_QUERY,
            num_pairs=query_batch_size,
        )

    def _get_query_item_batch(self):
        candidate_batch_size = (1 + self.negative_ratio) * self.batch_size
        query_batch_size = self.batch_size

        item_ids = self._get_item_ids(candidate_batch_size)
        queries = self._get_queries(query_batch_size)
        tokenized_ids, mask = self._get_tokenized_query(query_batch_size)
        return SageBatch(
            keys={
                EntityType.ITEM: item_ids,
                EntityType.SEARCH_QUERY: queries[:4],
            },
            tensor_feats={
                EntityType.ITEM: {
                    **self._get_multiple_embs(candidate_batch_size, "gs_v5", self._get_normalized_emb),
                    **self._get_multiple_embs(candidate_batch_size, "ue_v4", self._get_binary_emb),
                    "item_is_v2": self._get_normalized_emb(candidate_batch_size),
                    **self._get_text_features(candidate_batch_size, self.string_features[EntityType.ITEM]),
                    "id_hash": torch.tensor([hash(s) for s in item_ids], dtype=torch.int64),
                },
                EntityType.SEARCH_QUERY: {
                    "query_text_input_ids": tokenized_ids,
                    "query_text_attention_mask": mask,
                    "id_hash": torch.tensor([hash(s) for s in queries], dtype=torch.int64),
                },
            },
            task_name=TaskName.METRIC_LEARNING,
            candidate_entity_type=EntityType.ITEM,
            query_entity_type=EntityType.SEARCH_QUERY,
            num_pairs=query_batch_size,
        )

    def _create_batch(self) -> SageBatch:
        if self._candidate_entity_type == EntityType.SIGNATURE:
            return self._get_query_signature_batch()
        elif self._candidate_entity_type == EntityType.ITEM:
            return self._get_query_item_batch()
        elif self._candidate_entity_type == EntityType.SEARCH_QUERY:
            return self._get_query_query_batch()
        else:
            raise ValueError(f"Invalid candidate entity type: {self._candidate_entity_type}")


class SageMultiIterator(Iterator[List[SageBatch]], IterableDataset):
    """
    Defines an iterator that takes as input a list of iterators returning individual batches and concatenates them
    together to return a list of batches.
    """

    def __init__(self, iterators: List[Iterator[SageBatch]]):
        self.iterators = iterators

    def __iter__(self) -> Iterator[List[SageBatch]]:
        return self

    def __next__(self) -> List[SageBatch]:
        return [next(iterator) for iterator in self.iterators]
