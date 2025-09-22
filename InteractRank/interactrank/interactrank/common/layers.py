from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import math

import logging
import warnings
from collections import defaultdict
from operator import itemgetter

from interactrank.common.continuous import ONE_HUNDRED_YEARS_IN_SEC
from interactrank.common.continuous import AutoMLContinuousNormalization
from interactrank.common.dense_norm import BatchDenseNormalization
from interactrank.common.lazy_layernorm import LazyLayerNorm
from interactrank.data.stats_metadata import FeatureStatsMetadata
from interactrank.data.lw_features import Feature
from interactrank.common.types import FeatureType

if TYPE_CHECKING:
    from torch import nn


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

EMBEDDABLE_TYPES = {FeatureType.SPARSE_NUMERIC, FeatureType.CATEGORICAL, FeatureType.MULTI_CATEGORICAL}
DEFAULT_EMBEDDING_DIM = 16
def get_embedding_dimension(vocab: List[int]) -> int:
    return 8 + int(math.log10(len(vocab) + 1)) * 12

class FeatureGrouper:
    """
    Helper class to group features for Summarization and Latent Cross modules
    """

    def __init__(
        self,
        feature_map: Dict[str, Feature],
        emb_name_to_vocab: Dict[str, List[int]],
        emb_name_to_dim: Optional[Dict[str, int]] = None,
        features_to_skip: Optional[List[str]] = None,
    ):
        self.feature_map = feature_map
        self.emb_name_to_vocab = emb_name_to_vocab
        self.emb_name_to_dim = emb_name_to_dim or {}
        self.features_to_skip = [] if features_to_skip is None else features_to_skip

    @property
    def feature_shapes(self) -> Dict[str, int]:
        """
        Compute the shape of each feature in the feature_map. Only features with AutoMLMetadata signal_type and
        signal_group metadata fields are processed.
        :return: The dictionary of feature name to its shape.
        """
        if not hasattr(self, "_feature_shapes"):
            self._feature_shapes = {}
            for feat_name, feat in self.feature_map.items():
                metadata = feat.metadata
                if _is_typed_metadata(metadata):
                    self._feature_shapes[feat_name] = self._get_feat_shape(feat)
        return self._feature_shapes

    @property
    def feature_groups(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Group features based on their AutoML signal type and signal group respectively.
        :return: The nested dictionary of signal_type to signal_group to its list of features.
        """
        if not hasattr(self, "_feature_groups"):
            self._feature_groups: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
            for feat_name, feat in self.feature_map.items():
                if feat_name in self.features_to_skip:
                    continue
                metadata = feat.metadata
                if _is_typed_metadata(metadata):
                    signal_group = metadata.signal_group
                    signal_type = metadata.signal_type
                    self._feature_groups[signal_type][signal_group].append(feat_name)
        return self._feature_groups

    @property
    def summarization_groups(self) -> Dict[str, List[str]]:
        """
        Build the batch summarization module groups based on its feature_groups.
        :return: The dictionary of group to list of features for BatchSummarization.
        """
        if not hasattr(self, "_summarization_groups"):
            self._summarization_groups = {}
            for signal_type, signal_group_dict in self.feature_groups.items():
                for signal_group, feats in signal_group_dict.items():
                    self._summarization_groups[_get_summarizing_group_name(signal_group, signal_type)] = feats
        return self._summarization_groups

    @property
    def summarization_groups_shapes(self) -> Dict[str, int]:
        """
        Compute the output shapes of each summarization group.
        :return: The dictionary of group to its output shape.
        """
        if not hasattr(self, "_summarization_groups_shapes"):
            self._summarization_groups_shapes = {}
            for group, feats in self.summarization_groups.items():
                sample_feat = feats[0]
                self._summarization_groups_shapes[group] = self.feature_shapes[sample_feat]
        return self._summarization_groups_shapes

    @property
    def latent_cross_groups(self) -> Dict[str, List[str]]:
        """
        Build the batch latent cross module groups based on its feature_groups.
        :return: The dictionary of group to list of features for BatchLatentCross.
        """
        if not hasattr(self, "_latent_cross_groups"):
            self._latent_cross_groups = defaultdict(list)
            for signal_type, signal_group_dict in self.feature_groups.items():
                for signal_group in signal_group_dict.keys():
                    self._latent_cross_groups[signal_type].append(
                        _get_summarizing_group_name(signal_group, signal_type)
                    )
        return self._latent_cross_groups

    def _get_feat_shape(self, feature: Feature) -> int:
        signal_type: str = feature.metadata.signal_type
        if feature.definition.feature_type in EMBEDDABLE_TYPES:
                emb_name = "type_" + signal_type
                default_dim = (
                    get_embedding_dimension(self.emb_name_to_vocab[emb_name])
                    if emb_name in self.emb_name_to_vocab
                    else DEFAULT_EMBEDDING_DIM
                )
                return self.emb_name_to_dim.get(emb_name, default_dim)
        elif feature.definition.feature_type == FeatureType.DENSE_NUMERIC:
                return feature.definition.shape[0]

        raise ValueError(f"Cannot infer summarizing group shape for {feature.definition.name}.")


def _get_summarizing_group_name(signal_group: str, signal_type: str) -> str:
    return f"{signal_group}_#_{signal_type}".replace(".", "_")


def _is_typed_metadata(metadata) -> bool:
    return metadata is not None and metadata.signal_group is not None and metadata.signal_type is not None


def _vocab_name_for_signal_type(signal_type: str) -> str:
    """
    :param signal_type: The signal type
    :return: The embedding name for the signal type
    """
    return f"type_{signal_type}"


def _merge_vocab(first: Dict[int, int], second: Dict[int, int]) -> Dict[int, int]:
    """
    Merge the two vocabularies summing the count of vocab items
    :param first: The first vocabulary containing counts and ids
    :param second: The second vocabulary containing counts and ids
    :return: The merged vocabulary
    """
    result = first
    for k, v in second.items():
        if k in result:
            result[k] += v
        else:
            result[k] = v
    return result


def generate_embedding_vocabs(
    feature_map: Dict[str, Feature],
    shared_embedding_voc_min_count: Optional[Dict[str, int]] = None,
    features_to_skip: Optional[Union[str, Sequence[str]]] = None,
    features_to_include: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[Dict[str, str], Dict[str, List[int]]]:
    """
    Generates embedding and the corresponding vocabs for the features, merging the embeddings for features with the same
    signal type.
    :param feature_map: The feature map
    :param shared_embedding_voc_min_count: Map containing the minimum count (occurrences seen by feature estimator) to
                add the id into the embedding vocabulary for each signal type. The map must contain a ''default''
                minimum count if all signal types are not present in the map
    :param features_to_skip: Features in feature map that we ignore. Skipped features does not contribute to any shared
                vocab.
    :param features_to_include: If not None, only consider features in feature map that are in this set and not in features_to_skip
    :return: Tuple of dict mapping feature name to corresponding embedding name and dict containing the vocab for each
            embedding
    """

    if shared_embedding_voc_min_count is None:
        shared_embedding_voc_min_count = {"default": 1}
    feat_to_emb_name = {}
    emb_name_to_vocab = {}
    shared_emb_to_vocab = defaultdict(dict)
    for feature_name, feature in feature_map.items():
        if features_to_skip and feature_name in features_to_skip:
            continue
        if features_to_include and feature_name not in features_to_include:
            continue
        if feature.definition.feature_type in EMBEDDABLE_TYPES:
            automl_metadata = feature.metadata
            vocab = feature.metadata.vocab
            assert vocab is not None, f"vocab is none for {feature.definition.name}"
            if automl_metadata is not None and automl_metadata.signal_type is not None:
                shared_emb_to_vocab[automl_metadata.signal_type] = _merge_vocab(
                    shared_emb_to_vocab[automl_metadata.signal_type], vocab
                )
                feat_to_emb_name[feature_name] = _vocab_name_for_signal_type(automl_metadata.signal_type)
            else:
                vocab = [k for k, _ in sorted(vocab.items(), key=itemgetter(1), reverse=True)]
                LOG.debug(f"own embeddings {feature_name} : (min={min(vocab)}, max={max(vocab)}, vocsize={len(vocab)})")
                emb_name_to_vocab[feature_name] = vocab
                feat_to_emb_name[feature_name] = feature_name
    for signal_type, vocab in shared_emb_to_vocab.items():
        min_count = shared_embedding_voc_min_count.get(signal_type, shared_embedding_voc_min_count["default"])
        vocab = [k for k, cnt in sorted(vocab.items(), key=itemgetter(1), reverse=True) if cnt >= min_count]
        # raise an error if post-count-filter results in empty vocabularies
        if len(vocab) == 0:
            raise RuntimeError(
                f"shared embedding {signal_type} vocabulary is empty after filtering for count >= {min_count}"
            )
        else:
            LOG.debug(f"shared embeddings {signal_type} : (min={min(vocab)}, max={max(vocab)}, vocsize={len(vocab)})")
        emb_name_to_vocab[_vocab_name_for_signal_type(signal_type)] = vocab
    return feat_to_emb_name, emb_name_to_vocab


def create_automl_continuous_normalization_layer(
    feature_map: Dict[str, Feature],
    to_skip: Optional[Collection[str]] = None,
    include_timestamp_features: bool = False,
    explicit_scaling: bool = False,
    norm_layer_init_fn: Callable[[], nn.Module] = LazyLayerNorm,
    fill_nan_values: bool = False,
    use_mixture_transformation: bool = False,
) -> AutoMLContinuousNormalization:
    """
    Creates a AutoMLContinuousNormalization layer based on a MLEnvFeatureMap with already populated
    FEATURE_STATS metadata.

    :param feature_map: The MLEnvFeatureMap feature map.
    :param to_skip: An optional set of feature names to skip.
    :param include_timestamp_features: Flag to indicate if timestamp features are to be processed.
    :param explicit_scaling: apply 0-1 scaling after log scaling, to scale all feature to be between [0, 1]
    :param norm_layer_init_fn: The layer normalization function to apply at the end.
    :param fill_nan_values: Whether to fill nan values with 0.
    :param use_mixture_transformation: if True, we learn a mixture of transformations to be applied per numerical feature
            as proposed in https://research.google/pubs/pub49171/
    :return: An AutoMLContinuousNormalization layer with min/max/stddev buffers populated accordingly.
    """
    to_skip = to_skip or {}
    feature_stats_dict = {}
    for feature_name in feature_map:
        if feature_map[feature_name].definition.feature_type != FeatureType.NUMERIC or feature_name in to_skip:
            continue
        stat = feature_map[feature_name].metadata
        if (
            include_timestamp_features
        ):
            stat = FeatureStatsMetadata(min_value=0, max_value=ONE_HUNDRED_YEARS_IN_SEC, std=1, mean=0)
        feature_stats_dict[feature_name] = stat

    return create_automl_continuous_normalization_layer_from_feature_stats(
        feature_stats_dict,
        explicit_scaling=explicit_scaling,
        norm_layer_init_fn=norm_layer_init_fn,
        fill_nan_values=fill_nan_values,
        use_mixture_transformation=use_mixture_transformation,
    )


def create_automl_continuous_normalization_layer_from_feature_stats(
    feature_stats_dict: Dict[str, FeatureStatsMetadata], **kwargs
):
    """
    Creates a AutoMLContinuousNormalization layer based on a dict of feature name to FEATURE_STATS metadata.

    :param feature_stats_dict: Dict of feature name to FEATURE_STATS metadata
    :param kwargs: Extra keyword args for AutoMLContinuousNormalization
    :return: An AutoMLContinuousNormalization layer with min/max/stddev buffers populated accordingly.
    """
    min_values = []
    max_values = []
    mean_values = []
    std = []
    feats = []
    for feature_name, stat in feature_stats_dict.items():
        min_values.append(stat.min_value)
        max_values.append(stat.max_value)
        mean_values.append(stat.mean)
        std.append(stat.std)
        feats.append(feature_name)

    return AutoMLContinuousNormalization(
        feature_names=feats,
        min_values_stat=min_values,
        max_values_stat=max_values,
        mean_values_stat=mean_values,
        stddev_stat=std,
        **kwargs,
    )


def create_batch_dense_normalization(
    feature_map: Dict[str, Feature],
    to_skip: Optional[Collection[str]] = None,
    norm_layer_init_fn=LazyLayerNorm,
    dense_norm_fn=BatchDenseNormalization,
):
    """
    Creates a BatchDenseNormalization layer based on the feature map. Each feature that is FeatureType.DENSE_NUMERIC
    will be included in the normalization unless explicitly skipped.

    :param feature_map The MLEnvFeatureMap containing all the features used by the model.
    :param to_skip: List of features to skip normalization.
    :param norm_layer_init_fn: The normalization layer init function to apply.

    :return: A BatchDenseNormalization layer
    """
    to_skip = to_skip if to_skip else {}
    feats_to_normalize = []
    for feat_name, feat in feature_map.items():
        if feat.definition.feature_type == FeatureType.DENSE_NUMERIC and feat_name not in to_skip:
            feats_to_normalize.append(feat_name)

    return dense_norm_fn(feats_to_normalize, norm_layer_init_fn=norm_layer_init_fn)
