from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
import numpy as np
from interactrank.common.types import SearchLabels
from typing import Union

import random
import unicodedata

import torch
from interactrank.common.types import EntityType
from interactrank.data.lw_features import LwBatch
from interactrank.data.lw_features import TaskName
from torch.utils.data import IterableDataset

if TYPE_CHECKING:
    from torch import Tensor


class RandomDataset(Iterator[LwBatch], IterableDataset):
    def __init__(
        self,
        num_examples: Optional[int] = None,
        batch_size: int = 1024,
        negative_ratio: int = 4,
        query_vocab_size: int = 256_000,
        vocab_size: int = 10_000_000,
        candidate_entity_type: EntityType = EntityType.SIGNATURE,
    ) -> None:
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

    def __next__(self) -> LwBatch:
        if self._num_examples is not None and (self._idx * self.batch_size) >= self._num_examples:
            raise StopIteration
        self._idx += 1
        return self._batch.batch

    def check_batch(self):
        return self._batch
    def _get_normalized_emb(self, *sizes: int) -> torch.Tensor:
        return torch.nn.functional.normalize(torch.randn(*sizes, 64)).to(dtype=torch.float32)

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
    def get_item_ids(self, size: int) -> torch.Tensor:
        # Sample 48 length random strings consisting of all printable ascii characters
        item_ids = ["".join([chr(i) for i in torch.randint(32, 127, (48,))]) for _ in range(size)]
        hashes = np.array([(hash(s) % 256) - 128 for s in item_ids], dtype=np.int8)  # shape: (size,)
        return torch.frombuffer(hashes.tobytes(), dtype=torch.int8).reshape(size, -1)

    def get_cross_features(self, *sizes: int) -> torch.Tensor:
        return torch.rand(*sizes, dtype=torch.float32)

    def get_continous_features(self, *sizes: int) -> torch.Tensor:
        return torch.randint(0, 101, (*sizes,), dtype=torch.float32)

    def get_labels(self, size:int) -> torch.Tensor:
        return torch.randint(0, 2, (size,), dtype=torch.bool)

    def get_categorical_features(self, *sizes: int) -> torch.Tensor:
        # Generate random categorical values between 0 and 10 (assuming 11 categories)
        return torch.randint(0, 11, (*sizes,), dtype=torch.float32)

    def _unicode_fuzz(self, length: int) -> str:
        # generate it
        utf_string = ''.join([random.choice(self.unicode_glyphs) for _ in range(length)])
        return utf_string

    def _get_queries(self, size: int) -> List[str]:
        # sample lengths between 10 to 100
        lengths = torch.randint(10, 100, (size,))
        # sample random unicode strings according to the lengths
        return [self._unicode_fuzz(length) for length in lengths]

    def _get_query_signature_batch(self) -> LwBatch:
        candidate_batch_size = self.batch_size
        return LwBatch(
            batch={
                f"{EntityType.LABEL.value}/image_sig": self.get_item_ids(candidate_batch_size),
                f"{EntityType.SIGNATURE.value}/gs_v5": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SIGNATURE.value}/ue_v4": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SIGNATURE.value}/searchsage_item": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.CROSS.value}/item_save_90day": self.get_cross_features(candidate_batch_size),
                f"{EntityType.CROSS.value}/item_click_90day": self.get_cross_features(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/user_seq_action_type": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/user_seq_gs_v5": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/searchsage_query": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/user_seq_timestamp": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/timestamp": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/itemsage_seq": self._get_normalized_emb(candidate_batch_size),
                f"{EntityType.SIGNATURE.value}/sig_count": self.get_continous_features(candidate_batch_size),
                f"{EntityType.SEARCH_QUERY.value}/category_count": self.get_continous_features(candidate_batch_size),
                f"{EntityType.SIGNATURE.value}/item_category": self.get_categorical_features(candidate_batch_size),
                SearchLabels.REPIN.value: self.get_labels(candidate_batch_size),
                SearchLabels.CLICK.value: self.get_labels(candidate_batch_size),
                SearchLabels.CLOSEUP.value: self.get_labels(candidate_batch_size),
                SearchLabels.IMPRESSION.value: self.get_labels(candidate_batch_size),
                SearchLabels.LONG_CLICK.value: self.get_labels(candidate_batch_size),
                SearchLabels.REQUEST_ID.value: self.get_item_ids(candidate_batch_size),
                SearchLabels.USER_ID.value: self.get_item_ids(candidate_batch_size),
                SearchLabels.ITEM_ID.value: self.get_item_ids(candidate_batch_size),

            },
            task_name=TaskName.METRIC_LEARNING,
            candidate_entity_type=EntityType.SIGNATURE,
            query_entity_type=EntityType.SEARCH_QUERY,
        )

    def _create_batch(self) -> LwBatch:
        if self._candidate_entity_type == EntityType.SIGNATURE:
            return self._get_query_signature_batch()
        elif self._candidate_entity_type == EntityType.SEARCH_QUERY:
            return self._get_query_query_batch()
        else:
            raise ValueError(f"Invalid candidate entity type: {self._candidate_entity_type}")


class LwMultiIterator(Iterator[LwBatch], IterableDataset):
    """
    Defines an iterator that takes as input a list of iterators returning individual batches and concatenates them
    together to return a list of batches.
    """

    def __init__(self, iterators: Iterator[LwBatch]):
        self.iterators = iterators

    def __iter__(self) -> Iterator[LwBatch]:
        return self

    def __next__(self) -> LwBatch:
        return next(self.iterators)  # for iterator in self.iterators
