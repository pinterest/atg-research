from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Tuple

import itertools

import simplejson as json
import torch
from thrift_utils.utils import thrift_from_binary_data
from trainer.ppytorch.data.sage_features import SageBatch
from trainer.ppytorch.data.sage_features import SageFeaturizer
from trainer.ppytorch.data.sage_features import SageFeaturizerV2
from trainer.ppytorch.data.sage_features import TaskName
from trainer.ppytorch.data.sage_features import check_tensor_feats
from trainer.ppytorch.data.sage_features import filter_features
from trainer.ppytorch.data.ufr_features import FbsUFRBatchFeatureExtractor

from schemas.graphsage.ttypes import GraphSageMetadata
from schemas.graphsage_common.ttypes import EntityType

if TYPE_CHECKING:
    import pyarrow as pa
    from torch import nn
    from trainer.ppytorch.data.sage_features import RawSageBatch
    from trainer.ppytorch.data.sage_s3_db import RawIndexData
    from trainer.ppytorch.data.sage_s3_db import RawPairData


def extract_weights(metas: List[GraphSageMetadata]) -> Dict[str, torch.Tensor]:
    """
    Extract weights from GraphSageMetadatas.
    We should set all 4 of these fields in the training prep workflow,
    so we expect no missing values, but they're not correctly set for relevance
    datsets yet, hence the fallback to meta.info[volume] and 0
    """
    key_to_wt = {
        "action_ct": [],
        "query_action_ct": [],
        "cand_action_ct": [],
        "query_volume": [],
    }

    for meta in metas:
        for k, v in key_to_wt.items():
            meta_attr = getattr(meta, k, None)
            if meta_attr is None:
                # query_volume isn't correctly populated in relevance datasets
                # TODO(pagarwal): Remove once the old datasets with the old prod model are removed
                if k == "query_volume":
                    meta_attr = (meta.info or {}).get("volume")
                    meta_attr = meta_attr if meta_attr is not None else 0
                else:
                    meta_attr = 0
            assert meta_attr is not None, meta
            v.append(int(meta_attr))

    return {k: torch.tensor(v, dtype=torch.long) for k, v in key_to_wt.items()}


def extract_cand_features(
    raw_batch: RawSageBatch,
    entity_type: EntityType,
    string_features: List[str],
    ufr_extractor: FbsUFRBatchFeatureExtractor,
    tokenizer: nn.Module,
    query_text_feature_name: str,
) -> Tuple[list[str], Dict[str, torch.Tensor]]:
    """
    Utility method that takes the raw batch, entity_type, string_feature names, ufr_extractor and tokenizers
    and then returns the candidate list along with string/tensor features
    Args:
        raw_batch: raw sagebatch that is used to get the final trainable batch
        entity_type: entity type, e.g. SIGNATURE, ITEM etc.
        string_features: list of string features to be used in traning/eval
        ufr_extractor: extractor object to be used to extract features
        tokenizer: tokenizer to be used in tokenization for strings for training/eval
        query_text_feature_name: feature name for query text
    """
    cands, cand_ser_feats = filter_features(raw_batch, target_entity_type=entity_type)
    cand_feats = ufr_extractor.extract_batch(cand_ser_feats).features if cands and ufr_extractor else {}
    cand_feats["id_hash"] = torch.ops.pinterest_ops.hash_tokenize(cands)
    if entity_type == EntityType.SEARCH_QUERY:
        cand_feats[query_text_feature_name] = cands
    if cands and string_features and tokenizer:
        for field in string_features:
            tokenized_text = tokenizer(cand_feats.get(field))
            cand_feats.update({f"{field}_{k}": v for k, v in tokenized_text.items()})

    return cands, cand_feats


class SearchSageFeaturizerBase:
    """
    Featurizer for SearchSAGE training.
    String and tensor fields are specified separately because the c++ flatbuffer op
    returns (list[tensor], List[List[str]]), so to assign names to these we need
    to know which are strings and which are tensors
    """

    def __init__(
        self,
        tensor_feature_names: Dict[EntityType, List[str]],
        string_feature_names: Dict[EntityType, List[str]],
        tokenizers: Dict[EntityType, nn.Module] = None,
        query_text_feature_name: str = "query_text",
    ):
        """
        Args:
            tensor_feature_names: dict of entity type to names of fields holding tensors that should be extracted from input UFRs
            string_feature_names: dict of entity type to names of fields holding tensors that should be extracted from input UFRs
            tokenizers: tokenizer for different entity types
        """
        self.tensor_feature_names = tensor_feature_names
        self.string_feature_names = string_feature_names
        self.ufr_extractor = {}
        for entity_type in tensor_feature_names:
            # Since query_text is a virtual feature for Search query, remove from passign into ufr extractor
            string_features = string_feature_names[entity_type]
            if entity_type == EntityType.SEARCH_QUERY:
                string_features = [feature for feature in string_features if feature != query_text_feature_name]
            if tensor_feature_names[entity_type] or string_features:
                self.ufr_extractor[entity_type] = FbsUFRBatchFeatureExtractor(
                    feature_names=self.tensor_feature_names[entity_type],
                    string_feat_names=self.string_feature_names[entity_type],
                )
        self.tokenizers = tokenizers or {}
        self.query_text_feature_name = query_text_feature_name

    def __repr__(self) -> str:
        attrs = {
            "class_name": self.__class__.__name__,
            "tensor_feats": {EntityType._VALUES_TO_NAMES[k]: v for k, v in self.tensor_feature_names.items()},
            "string_feats": {EntityType._VALUES_TO_NAMES[k]: v for k, v in self.string_feature_names.items()},
        }
        return json.dumps(attrs, sort_keys=True, indent=2)


class SearchSageFeaturizer(SageFeaturizer, SearchSageFeaturizerBase):
    def triplet_collate_fn(
        self,
        query_result_batch: List[RawPairData],
        index_batch: List[RawIndexData],
    ) -> SageBatch:
        """
        Extracts a minibatch of triplets <query, positive, negative> from a sequence of RawPairData
        and a sequence of RawIndexData.

        In the extraction logic, queries will be followed by positives and then by negatives. The order will be
        preserved in the final SageBatch if the queries, positives or negatives are entities of the same type.
        """
        query_batch = ((row.query_key, row.query_entity_type, row.feat1, row.meta) for row in query_result_batch)
        result_batch = ((row.cand_key, row.cand_entity_type, row.feat2, None) for row in query_result_batch)
        index_batch = ((row.key, row.entity_type, row.feat, row.meta) for row in index_batch)
        batch = self.extract_batch(
            list(itertools.chain.from_iterable((query_batch, result_batch, index_batch))),
            num_pairs=len(query_result_batch),
        )
        batch.query_entity_type = query_result_batch[0].query_entity_type
        batch.candidate_entity_type = query_result_batch[0].cand_entity_type
        batch.num_pairs = len(query_result_batch)
        return batch

    def pair_collate_fn(self, query_result_batch: List[RawPairData]) -> SageBatch:
        """
        Extracts a minibatch of <query, positive> pairs from a sequence of RawPairData.

        In the extraction logic, all the queries will precede the positive entities. The order will be preserved
        in the final SageBatch if queries and positives are entities of the same type.
        """
        query_batch = ((row.query_key, row.query_entity_type, row.feat1, row.meta) for row in query_result_batch)
        result_batch = ((row.cand_key, row.cand_entity_type, row.feat2, None) for row in query_result_batch)
        batch = self.extract_batch(
            list(itertools.chain.from_iterable((query_batch, result_batch))), num_pairs=len(query_result_batch)
        )
        batch.query_entity_type = query_result_batch[0].query_entity_type
        batch.candidate_entity_type = query_result_batch[0].cand_entity_type
        batch.num_pairs = len(query_result_batch)
        return batch

    def extract_batch(self, raw_batch: RawSageBatch, num_pairs: int = None) -> SageBatch:
        # pin/item feature processing
        all_keys = {}
        texts = {}
        all_tensor_feats = {}
        for entity_type in self.tensor_feature_names:
            cand_ids, cand_feats = extract_cand_features(
                raw_batch=raw_batch,
                entity_type=entity_type,
                string_features=self.string_feature_names[entity_type],
                ufr_extractor=self.ufr_extractor.get(entity_type),
                tokenizer=self.tokenizers.get(entity_type),
                query_text_feature_name=self.query_text_feature_name,
            )
            if cand_ids:
                all_keys[entity_type] = cand_ids
                if not self.tokenizers:  # this is the case when we dont tokenize the strings
                    all_tensor_feats[entity_type] = cand_feats
                else:  # this is the case when we tokenize all strings
                    texts.update({field: cand_feats.pop(field) for field in self.string_feature_names[entity_type]})
                    all_tensor_feats[entity_type] = check_tensor_feats(cand_feats)

        # query metadata processing
        meta_bins = [meta for _, entity_type, _, meta in raw_batch[:num_pairs]] if num_pairs else []
        metas = [thrift_from_binary_data(GraphSageMetadata, meta_bin) for meta_bin in meta_bins]
        weights = extract_weights(metas)

        # batch creation
        batch = SageBatch(
            keys=all_keys,
            tensor_feats=all_tensor_feats,
            texts=texts,
            weights=weights,
            metas=metas,
            task_name=TaskName.METRIC_LEARNING,
        )
        return batch


class SearchSageFeaturizerV2(SageFeaturizerV2, SearchSageFeaturizerBase):
    def extract_batch(
        self, all_feats: Dict[EntityType, pa.Array], all_ids: Dict[EntityType, List[str]], meta_bins: List[bytes] = None
    ):
        all_keys = {}
        all_tensor_feats = {}
        texts = {}
        for entity_type in all_feats:
            string_features = self.string_feature_names[entity_type]
            tokenizer = self.tokenizers.get(entity_type)

            feats = all_feats[entity_type]
            cand_ids = all_ids[entity_type]
            ufr_extractor = self.ufr_extractor.get(entity_type)
            cand_feats = ufr_extractor.extract_batched_pyarrow(feats) if ufr_extractor else {}

            cand_feats["id_hash"] = torch.ops.pinterest_ops.hash_tokenize(cand_ids)

            if entity_type == EntityType.SEARCH_QUERY:
                cand_feats[self.query_text_feature_name] = cand_ids
            if string_features and tokenizer:
                for field in string_features:
                    tokenized_text = tokenizer(cand_feats.get(field))
                    cand_feats.update({f"{field}_{k}": v for k, v in tokenized_text.items()})
            all_keys[entity_type] = cand_ids
            if not self.tokenizers:  # this is the case when we dont tokenize the strings
                all_tensor_feats[entity_type] = cand_feats
            else:  # this is the case when we tokenize all strings
                texts.update({field: cand_feats.pop(field) for field in self.string_feature_names[entity_type]})
                all_tensor_feats[entity_type] = check_tensor_feats(cand_feats)

        # query metadata processing
        metas = [thrift_from_binary_data(GraphSageMetadata, meta_bin) for meta_bin in meta_bins]
        weights = extract_weights(metas)

        # batch creation
        batch = SageBatch(
            keys=all_keys,
            tensor_feats=all_tensor_feats,
            texts=texts,
            weights=weights,
            metas=metas,
            task_name=TaskName.METRIC_LEARNING,
        )
        return batch
