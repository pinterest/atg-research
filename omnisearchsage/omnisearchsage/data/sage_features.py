from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import dataclasses
import enum
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import torch
from omnisearchsage.common.types import EntityType


def _apply_func(func: Callable[[torch.Tensor], torch.Tensor], data: Any, ignore_error: bool = False):
    """Recursively move tensors to the device"""
    if isinstance(data, pa.Array) or isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return func(data)

    if isinstance(data, list):
        if data and isinstance(data[0], (str, bytes)):
            return data
        return [_apply_func(func, v, ignore_error) for v in data]

    if isinstance(data, tuple):
        return tuple(_apply_func(func, v, ignore_error) for v in data)

    if isinstance(data, dict):
        return {k: _apply_func(func, v, ignore_error) for k, v in data.items()}

    if dataclasses.is_dataclass(data):
        args = {}
        for f in dataclasses.fields(data):
            args[f.name] = _apply_func(func, getattr(data, f.name), ignore_error)
        return type(data)(**args)

    if ignore_error:
        return data
    else:
        raise NotImplementedError(f"Unsupported type in tensor dict: {type(data)}")


def tensor_to_nonblocking(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)


def send_to_device(data, device: Union[torch.device, str], ignore_error: bool = False):
    """Recursively move tensors to the device"""
    if isinstance(device, str):
        device = torch.device(device)
    return _apply_func(lambda x: tensor_to_nonblocking(x, device), data=data, ignore_error=ignore_error)


def pin_memory(data, device: Union[torch.device, str], ignore_error: bool = False):
    """Recursively move tensors to the device"""
    return _apply_func(lambda x: x.pin_memory(device), data=data, ignore_error=ignore_error)


class TaskName(enum.IntEnum):
    METRIC_LEARNING = 1


@dataclass
class SageBatch(object):
    """A minibatch of features using to for training and evaluating SAGE models."""

    keys: Dict[EntityType, List[str]]
    tensor_feats: Dict[EntityType, Dict[str, torch.Tensor]]
    task_name: TaskName
    texts: Optional[Dict[str, List[str]]] = None
    query_entity_type: Optional[EntityType] = None
    candidate_entity_type: Optional[EntityType] = None
    num_pairs: Optional[int] = None

    def pin_memory(self, device=None):
        return pin_memory(self, device, ignore_error=True)

    @property
    def sigs(self) -> Optional[List[str]]:
        return self.keys.get(EntityType.SIGNATURE)

    @property
    def search_queries(self) -> Optional[List[str]]:
        return self.keys.get(EntityType.SEARCH_QUERY)

    @property
    def item_ids(self) -> Optional[List[str]]:
        return self.keys.get(EntityType.ITEM)

    @property
    def sig_feats(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.tensor_feats.get(EntityType.SIGNATURE)

    @property
    def search_query_feats(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.tensor_feats.get(EntityType.SEARCH_QUERY)

    @property
    def item_feats(self) -> Optional[Dict[str, torch.Tensor]]:
        return self.tensor_feats.get(EntityType.ITEM)

    @property
    def entity_types(self) -> List[EntityType]:
        if not self.keys:
            return []

        return [entity_type for entity_type, keys in self.keys.items() if keys]

    def replace(self, **attrs) -> SageBatch:
        return dataclasses.replace(self, **attrs)


def check_tensor_feats(tensor_feats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Ensures that the provided dictionary of features contains only tensor features.
    """
    for k, v in tensor_feats.items():
        assert isinstance(v, torch.Tensor), f"{k} {v}"

    return tensor_feats


class SageFeaturizerV2:
    def extract_batch(
        self, all_feats: Dict[EntityType, pa.Array], all_ids: Dict[EntityType, List[str]], meta_bins: List[bytes] = None
    ):
        """
        Extracts a minibatch of features from a sequence of RawSageBatch.

        The extracted features will be stored in a SageBatch object.
        """
        raise NotImplementedError()

    @staticmethod
    def _extract_feats_and_ids(
        query_entity_type: EntityType,
        candidate_entity_type: EntityType,
        pairs: pa.Table,
        index: Optional[pa.Table] = None,
    ):
        feats = defaultdict(list)
        ids = defaultdict(list)

        feats[query_entity_type].extend(pairs["feat1"].chunks)
        ids[query_entity_type].extend(pairs["query_key"].chunks)
        feats[candidate_entity_type].extend(pairs["feat2"].chunks)
        ids[candidate_entity_type].extend(pairs["cand_key"].chunks)
        if index is not None:
            feats[candidate_entity_type].extend(index["feat"].chunks)
            ids[candidate_entity_type].extend(index["key"].chunks)

        return (
            {entity_type: pa.concat_arrays(feats[entity_type]) for entity_type in feats},
            {entity_type: pa.concat_arrays(ids[entity_type]).to_pylist() for entity_type in ids},
        )

    def triplet_collate_fn(self, pairs: pa.Table, index: pa.Table) -> SageBatch:
        # ["query_key", "cand_key", "query_entity_type", "cand_entity_type", "feat1", "feat2", "meta"]
        # ["key", "entity_type", "feat", "meta"]
        query_entity_type = pairs["query_entity_type"][0].as_py()
        candidate_entity_type = pairs["cand_entity_type"][0].as_py()
        index_entity_type = index["entity_type"][0].as_py()
        assert candidate_entity_type == index_entity_type, (candidate_entity_type, index_entity_type)

        all_feats, all_ids = self._extract_feats_and_ids(
            query_entity_type=query_entity_type,
            candidate_entity_type=candidate_entity_type,
            pairs=pairs,
            index=index,
        )
        meta_bins = pairs["meta"].to_pylist()
        batch = self.extract_batch(all_feats, all_ids, meta_bins)
        batch.query_entity_type = query_entity_type
        batch.candidate_entity_type = candidate_entity_type
        batch.num_pairs = len(pairs)
        return batch

    def single_collate_fn(self, index: pa.Table) -> SageBatch:
        """
        Extracts a minibatch negative entities from given list of RawIndexData
        """
        # ["key", "entity_type", "feat", "meta"]
        entity_type = index["entity_type"][0].as_py()
        feats = {entity_type: pa.concat_arrays(index["feat"].chunks)}
        ids = {entity_type: pa.concat_arrays(index["key"].chunks).to_pylist()}
        return self.extract_batch(feats, ids, None)

    def pair_collate_fn(self, pairs: pa.Table) -> SageBatch:
        # ["query_key", "cand_key", "query_entity_type", "cand_entity_type", "feat1", "feat2", "meta"]
        query_entity_type = pairs["query_entity_type"][0].as_py()
        candidate_entity_type = pairs["cand_entity_type"][0].as_py()
        all_feats, all_ids = self._extract_feats_and_ids(
            query_entity_type=query_entity_type,
            candidate_entity_type=candidate_entity_type,
            pairs=pairs,
        )
        meta_bins = pairs["meta"].to_pylist()
        batch = self.extract_batch(all_feats, all_ids, meta_bins)
        batch.query_entity_type = query_entity_type
        batch.candidate_entity_type = candidate_entity_type
        batch.num_pairs = len(pairs)
        return batch
