from __future__ import annotations

from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch import nn
from interactrank.common.utils.utils import ExampleType
from interactrank.common.mask_net import LazyParallelMaskNetLayers
from interactrank.common.layers import FeatureGrouper
from interactrank.common.layers import create_automl_continuous_normalization_layer
from interactrank.common.layers import create_batch_dense_normalization
from interactrank.data.lw_features import Feature

from interactrank.common.fully_connected_layers import FullyConnectedLayers
from interactrank.common.lazy_concat import LazyConcatInput
from interactrank.common.lazy_layernorm import LazyLayerNorm
from interactrank.common.summarization import BatchMLPSummarization
from interactrank.base_model import TwoTowerModel
from interactrank.common.timestamp import UserSeqTimestampEncoder
from interactrank.common.utils.torchscript_converter import TorchScriptDeployInfo

from interactrank.constants.base_constants import CANDIDATE_SIG_ITEM_FEATURE_GROUP_NAMES
from interactrank.constants.base_constants  import CONTEXT_FEATURE_GROUP_NAMES
from interactrank.constants.base_constants  import DEFAULT_TIMESTAMP_FEAT_NAME
from interactrank.constants.base_constants  import MULTI_HEAD_WEIGHTS_FEATURE
from interactrank.constants.base_constants  import CROSS_FEATURES
from interactrank.configs.base_configs import NUM_HEADS
from interactrank.constants.base_constants  import OUTPUT_NAMES
from interactrank.constants.base_constants  import DOT_PRODUCT_FEATURE_FIELD
from interactrank.common.types import EntityType

LW_USER_SEQUENCE_CONFIG = {
    "algo": "average",  # can be one of 'mlp', 'average'
    "gsv5_ft_name": f"{EntityType.SIGNATURE.value}/gs_v5",
    "seq_action_ft_name": f"{EntityType.SEARCH_QUERY.value}/user_seq_action_type",
    "seq_emb_ft_name": f"{EntityType.SEARCH_QUERY.value}/user_seq_gs_v5",
    "seq_timestamp_ft_name": f"{EntityType.SEARCH_QUERY.value}/user_seq_timestamp",
    "request_time_ft_name": f"{EntityType.SEARCH_QUERY.value}/timestamp",
    "is_random_time_window": False,
    "time_window_ms": 1000 * 60 * 60 * 24 * 1,  # in ms
    "concat_max_pool": True,
    "latest_n_emb": 10,
    "seq_len": 100,
    "user_seq_time_window_in_day": 1,
    "seq_pinnersage_v3e_ft_name": f"{EntityType.SEARCH_QUERY.value}/itemsage_seq",
}

LW_QUERY_SEQ_CONFIG = {
    "query_seq_len": 20,
    "query_seq_navboost_embedding_ft_name": f"{EntityType.SEARCH_QUERY}/search_query_item_xperf_seq",
}

PMN_CONFIG = {
    # Parallel mask net params
    "pmn_output_dim_pin": 3072,
    "pmn_output_dim_query": 2048,
    "pmn_project_ratio": 0.1,
    "pmn_dropout_ratio": 0.005,
    "pmn_block_num": 4,
}

SEARCHSAGE_FEATURES = {
    "query": f"{EntityType.SEARCH_QUERY.value}/searchsage_query",
    "item": f"{EntityType.SIGNATURE.value}/searchsage_item",
}
SEARCHSAGE_EMBED_FEATURE = "searchsage"

class AverageNormalization(nn.Module):
    keeping_mask: torch.Tensor

    def __init__(
        self,
        feature_names: List[str],
    ):
        super().__init__()
        self.feature_names = feature_names

    def compute_average(self, tensor: torch.Tensor) -> torch.Tensor:
        avg_value = torch.mean(tensor)
        tensor[tensor == 0] = avg_value
        return tensor

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_tensor_dict = {k: v for k, v in tensor_dict.items()}
        for i, feature_name in enumerate(self.feature_names):
            tensor = tensor_dict[feature_name]
            assert tensor.dim() == 1
            out_tensor_dict[feature_name] = self.compute_average(tensor)

        return out_tensor_dict


class LpNormalize(nn.Module):
    def __init__(self, p: float = 2.0, num_heads: int = 1, embedding_dim: int = 64):
        super().__init__()
        self.p = p
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim

    def forward(self, x):
        return torch.nn.functional.normalize(x.view(-1, self.embedding_dim), p=2, dim=-1).view(x.shape)

    def extra_repr(self) -> str:
        return f"p={self.p}"


"""
This layer takes the embedding as input and returns sigmoid output as score
"""


class LpScore(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sigmoid(torch.sum(x, dim=1) + self.global_bias)


class Tower(nn.Module):
    def __init__(
        self,
        features: List[str],
        feature_map: Dict[str, Feature],
        embedding_layer: nn.Module,
        hidden_sizes: List[int],
        emb_name_to_vocab: Dict[str, List[int]],
        embedding_dim: int,
        time_feature: str,
        enable_user_sequence: bool,
        tower_type: str,
        use_skip_connections: bool = False,
        enable_pmn: bool = False,
        num_heads: int = 1,
    ):
        """
        :param feature_map: a feature map for a tower
        :param embedding_layer: a shared embedding layer for two towers
        :param hidden_sizes: a list of integer representing the hidden sizes
        :param emb_name_to_vocab: a dictionary of str to List
        :param embedding_dim: integer, the final embedding dim for the tower
        :param time_feature: name of time feature
        :param enable_continuous_layer: enable continuous layer or not
        :param enable_static_rank: enable static rank
        :param enable_user_sequence: use user sequence as feature when training the model
        :param enable_visual_uve4_pinnersage_v3e: use unified embedding as feature on pin side when training the model
        """
        super().__init__()
        feature_grouper = FeatureGrouper(feature_map, emb_name_to_vocab=emb_name_to_vocab)
        self.searchsage_feature = SEARCHSAGE_FEATURES[tower_type]
        self.searchsage_embedder = nn.Sequential(FullyConnectedLayers(32, [256]))
        # update feature map based on feature names passed in the tower
        updated_feature_map = feature_map
        layers = [
                create_automl_continuous_normalization_layer(
                    updated_feature_map,
                    to_skip={time_feature},
                    explicit_scaling=True,
                    norm_layer_init_fn=nn.Identity,
                    use_mixture_transformation=True,
                ),
                create_batch_dense_normalization(
                    updated_feature_map,
                    to_skip=[
                        f"{EntityType.SEARCH_QUERY}/user_seq_timestamp"
                    ],
                    norm_layer_init_fn= LazyLayerNorm,
                ),
                SearchSageLayer(self.searchsage_feature, self.searchsage_embedder),
                (
                    SmartEmbeddingLayer(feature_map, embedding_layer, emb_name_to_vocab)
                    if enable_user_sequence
                    else embedding_layer
                ),
                BatchMLPSummarization(
                    group_to_feats=feature_grouper.summarization_groups,
                    group_to_out_dim=feature_grouper.summarization_groups_shapes,
                ),
                LazyConcatInput(),
                LazyLayerNorm(),
                (
                    LazyParallelMaskNetLayers(
                        output_dim=(
                            PMN_CONFIG["pmn_output_dim_pin"]
                            if tower_type == "pin"
                            else PMN_CONFIG["pmn_output_dim_query"]
                        ),
                        project_ratio=PMN_CONFIG["pmn_project_ratio"],
                        dropout_ratio=PMN_CONFIG["pmn_dropout_ratio"],
                        block_num=PMN_CONFIG["pmn_block_num"],
                    )
                    if enable_pmn
                    else nn.Identity()
                ),
                FullyConnectedLayers(
                    output_size=embedding_dim * num_heads, hidden_sizes=hidden_sizes, has_skip=use_skip_connections
                ),
                LpNormalize(2.0, num_heads=num_heads, embedding_dim=embedding_dim),
            ]
        self.layers = nn.Sequential(*layers)
        self.embedding_dim = embedding_dim

    def forward(self, formatted_data) -> nn.Module:
        """
        Forward layer of the model
        :param formatted_data: The input batch of data.
        :return:final layer output
        """
        return self.layers(formatted_data)


class SearchSageLayer(nn.Module):
    def __init__(self, searchsage_feature, embedder) -> None:
        super().__init__()
        self.searchsage_feature = searchsage_feature
        self.embedder = embedder

    def forward(self, formatted_data):
        formatted_data[SEARCHSAGE_EMBED_FEATURE] = self.embedder(formatted_data[self.searchsage_feature])
        return formatted_data


class SmartEmbeddingLayer(nn.Module):
    """
    Embedding layer which encapsulates the normal embedding layer along with user seqeunce features

    :param feature_map: MLEnvFeatureMap
    :param embedding_layer: the embedding layer for categorical features
    :param user_seq_config: LW_USER_SEQUENCE_CONFIG
    """

    def __init__(
        self,
        feature_map: Dict[str, Feature],
        embedding_layer,
        emb_name_to_vocab,
        user_seq_config=LW_USER_SEQUENCE_CONFIG,
        query_seq_config=LW_QUERY_SEQ_CONFIG,
    ) -> None:
        super().__init__()
        self.feature_map = feature_map
        self.embedding_layer = embedding_layer
        self.user_seq_config = user_seq_config
        self.query_seq_config = query_seq_config
        self.seq_action_ft_name = self.user_seq_config["seq_action_ft_name"]
        self.seq_pinnersage_embedding_ft_name = self.user_seq_config["seq_pinnersage_v3e_ft_name"]
        self.user_seq_layer = UserSequenceLayer(
            algo=self.user_seq_config["algo"],
            feature_map=feature_map,
            embedding_layer=embedding_layer,
            emb_name_to_vocab=emb_name_to_vocab,
            user_seq_config=user_seq_config,
            query_seq_config=query_seq_config,
        )

    def forward(self, formatted_data):
        return self.user_seq_layer(formatted_data)


class UserSequenceLayer(nn.Module):
    def __init__(
        self,
        algo: str,
        feature_map: Dict[str, Feature],
        embedding_layer,
        emb_name_to_vocab,
        user_seq_config,
        query_seq_config,
    ) -> None:
        super().__init__()
        assert algo in ["average", "transformer", "mlp"]
        self.algo = algo
        self.feature_map = feature_map
        self.embedding_layer = embedding_layer
        self.emb_name_to_vocab = emb_name_to_vocab
        self.user_seq_config = user_seq_config
        self.query_seq_config = query_seq_config
        self.seq_emb_ft_name = self.user_seq_config["seq_emb_ft_name"]
        self.seq_action_ft_name = self.user_seq_config["seq_action_ft_name"]
        self.seq_query_navboost_embedding_ft_name = self.query_seq_config["query_seq_navboost_embedding_ft_name"]
        self.seq_pinnersage_embedding_ft_name = self.user_seq_config["seq_pinnersage_v3e_ft_name"]
        if self.algo == "mlp" and self.seq_emb_ft_name in self.feature_map:
            self.init_mlp_layer()
        self.normalize_layer = LpNormalize(2.0)
        self.time_embedder = UserSeqTimestampEncoder(bucket_len=14)
        self.time_linear = nn.Linear(self.time_embedder.output_dim, 32)
        # self.time_norm = LazyLayerNorm()

    def init_mlp_layer(self):
        # This layer does weighted averaging over the user sequence embeddings while also jointly learning the weights
        # (w1 * seq1) + (w2 * seq2) + .. + (wn * seqn) where w1, w2, .., wn are scalar weights learned by this layer
        self.seq_mlp_layer_action_type = torch.nn.Conv1d(
            in_channels=self.user_seq_config["seq_len"], out_channels=1, kernel_size=1
        )
        self.seq_mlp_layer_pin_emb = torch.nn.Conv1d(
            in_channels=self.user_seq_config["seq_len"], out_channels=1, kernel_size=1
        )
        self.seq_mlp_layer_query_emb = torch.nn.Conv1d(
            in_channels=self.query_seq_config["query_seq_len"], out_channels=1, kernel_size=1
        )

    def forward(self, formatted_data):
        if self.seq_emb_ft_name in self.feature_map:
            if self.algo in ("mlp", "average"):
                # flatten user action sequence for embedding layer since it needs all tensors to be 1-D
                user_seq_shape = formatted_data[self.seq_action_ft_name].shape
                formatted_data[self.seq_action_ft_name] = formatted_data[self.seq_action_ft_name].flatten()
                formatted_data = self.embedding_layer(formatted_data)
                # embedding layer will flatten and then embed categoricals, so reshape back to original shape
                formatted_data[self.seq_action_ft_name] = formatted_data[self.seq_action_ft_name].reshape(
                    (*user_seq_shape, -1)
                )
                if self.algo == "average":
                    # average over all user sequence action type embeddings
                    formatted_data[self.seq_action_ft_name] = torch.mean(formatted_data[self.seq_action_ft_name], dim=1)
                    # average over all user sequence pin embeddings
                    formatted_data[self.seq_emb_ft_name] = torch.mean(formatted_data[self.seq_emb_ft_name], dim=1)
                elif self.algo == "mlp":
                    # add timestamp embedding to final sequence.
                    request_timestamp = formatted_data.pop(DEFAULT_TIMESTAMP_FEAT_NAME)
                    seq_timestamps = formatted_data.pop(self.user_seq_config["seq_timestamp_ft_name"])
                    request_time = request_timestamp.unsqueeze(-1).expand(-1, seq_timestamps.shape[1])
                    request_time = torch.mul(request_time, 1000.0).long()
                    # Create masked array where seq_timestamps equal -1
                    trivial_value_mask = seq_timestamps == -1
                    request_time.masked_fill_(trivial_value_mask.bool(), -1)
                    ts_emb = self.time_embedder(request_time - seq_timestamps.long())

                    # combine timestamp embedding with sequence embedding
                    formatted_data[self.seq_emb_ft_name] += ts_emb
                    formatted_data[self.seq_pinnersage_embedding_ft_name] += ts_emb

                    # handle action type sequence
                    formatted_data[self.seq_action_ft_name] = self.seq_mlp_layer_action_type(
                        formatted_data[self.seq_action_ft_name]
                    ).squeeze(1)

                    # handle gsv5 sequence
                    searchsage_weighted_gsv5_seq_emb = torch.einsum(
                        "ij,ijk->ik",
                        nn.functional.softmax(
                            torch.einsum(
                                "ik,ijk->ij",
                                formatted_data[SEARCHSAGE_EMBED_FEATURE],
                                formatted_data[self.seq_emb_ft_name],
                            ),
                            dim=1,
                        ),
                        formatted_data[self.seq_emb_ft_name],
                    )
                    formatted_data[self.seq_emb_ft_name] = (
                        self.seq_mlp_layer_pin_emb(formatted_data[self.seq_emb_ft_name]).squeeze(1)
                        + searchsage_weighted_gsv5_seq_emb
                    )
                    # handle navboost embedding
                    searchsage_weighted_ps_seq_emb = torch.einsum(
                        "ij,ijk->ik",
                        nn.functional.softmax(
                            torch.einsum(
                                "ik,ijk->ij",
                                formatted_data[SEARCHSAGE_EMBED_FEATURE],
                                formatted_data[self.seq_query_navboost_embedding_ft_name],
                            ),
                            dim=1,
                        ),
                        formatted_data[self.seq_query_navboost_embedding_ft_name],
                    )
                    formatted_data[self.seq_query_navboost_embedding_ft_name] = (
                        self.seq_mlp_layer_query_emb(formatted_data[self.seq_query_navboost_embedding_ft_name]).squeeze(
                            1
                        )
                        + searchsage_weighted_ps_seq_emb
                    )
                    # handle pinnersage sequence
                    searchsage_weighted_ps_seq_emb = torch.einsum(
                        "ij,ijk->ik",
                        nn.functional.softmax(
                            torch.einsum(
                                "ik,ijk->ij",
                                formatted_data[SEARCHSAGE_EMBED_FEATURE],
                                formatted_data[self.seq_pinnersage_embedding_ft_name],
                            ),
                            dim=1,
                        ),
                        formatted_data[self.seq_pinnersage_embedding_ft_name],
                    )
                    formatted_data[self.seq_pinnersage_embedding_ft_name] = (
                        self.seq_mlp_layer_pin_emb(formatted_data[self.seq_pinnersage_embedding_ft_name]).squeeze(1)
                        + searchsage_weighted_ps_seq_emb
                    )
            elif self.algo == "transformer":
                formatted_data = self.embedding_layer(formatted_data)
                formatted_data = self.seq_tfmr_layer(formatted_data, is_train=self.training)
        else:
            formatted_data = self.embedding_layer(formatted_data)

        return formatted_data


class CrossLayer(nn.Module):
    def __init__(
        self,
        features: List[str],
        feature_map: Dict[str, Feature],
        enable_compute_average_navboost: bool,
    ):
        """
        This layer takes in the dot product from two tower(query&pin) & the cross features to generate the final scores
        :param feature_map: a ML env feature map for a tower
        :param enable_compute_average_navboost: replace navboost features where 0 with average value of that feature column
        This tower takes in the dot product and concat it with cross features to get the final scores by passing it to
        through layers.
        """
        super().__init__()
        self.identity = nn.Identity()
        self.concat = LazyConcatInput(to_skip={DOT_PRODUCT_FEATURE_FIELD})
        self.enable_compute_average_navboost = enable_compute_average_navboost
        if self.enable_compute_average_navboost:
            # map cross features mentioned here with one defined in features_const.py
            self.compute_average = AverageNormalization(feature_names=CROSS_FEATURES)
        # other parameters
        print("Check cross fmap: ", feature_map)
        self.feature_map = feature_map
        self.features= features
        self.cross_feature_keys = self.feature_map.keys()
        self.total_cross_features = len(features)
        print("Check features: ", self.cross_feature_keys)
        self.linear = nn.Linear(self.total_cross_features+ 1, 1)
        # we initialize the weights to a constant because we want the cross layer weights on the dot product to be +ve,
        # which are sensitive to +ve init. This dependency is due to a Manas side requirement
        # according to @bjuneja's past experiments
        nn.init.constant_(self.linear.weight, 1.0)

    def get_features(self) -> List[str]:
        """
        :return: the feature map for the tower
        """
        return self.features

    def get_cross_feature_keys(self):
        """
        :return: the cross feature key set
        """
        return self.cross_feature_keys

    def get_summary_metrics(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Create the dictionary of summary metrics given the results from the forward pass
        :param inputs: dict containing data passed in loss_metrics
        :return dict of summary metrics
        """
        return {f"cross_layer_weight_{i}": weight.item() for i, weight in enumerate(self.linear.weight.reshape(-1))}

    def forward(self, formatted_data: Dict[str, torch.Tensor]) -> nn.Module:
        """
        Forward layer of the model
        :param formatted_data: The input batch of data.
        :return:final layer output
        """
        output = self.identity(formatted_data)
        if self.enable_compute_average_navboost:
            output = self.compute_average(output)
        output = torch.cat(
            [
                self.concat(output).unsqueeze(1).expand(-1, NUM_HEADS, -1),
                output[DOT_PRODUCT_FEATURE_FIELD].unsqueeze(2),
            ],
            dim=2,
        )

        output = self.linear(output).squeeze()
        return output


class MultiHeadContextTower(nn.Module):
    def __init__(self, context_tower: nn.Module, num_heads: int = NUM_HEADS):
        super().__init__()
        self.context_tower = context_tower
        # self.format_data_layer = format_data_layer
        self.num_heads = num_heads
        self.embedding_dim = context_tower.embedding_dim

    def forward(self, formatted_data) -> nn.Module:
        """
        Forward layer of the model
        :param formatted_data: The input batch of data.
        :return:final layer output
        """
        multi_head_weights = formatted_data.pop(MULTI_HEAD_WEIGHTS_FEATURE).float()
        multi_head_weights /= torch.sum(multi_head_weights, dim=1).unsqueeze(1)
        assert multi_head_weights.sum(dim=1).allclose(
            torch.ones(multi_head_weights.shape[0], device=multi_head_weights.device)
        )
        #formatted_data = self.format_data_layer(formatted_data)
        output = self.context_tower(formatted_data)
        output = torch.einsum("ijk,ij->ik", output.view(-1, self.num_heads, self.embedding_dim), multi_head_weights)
        return output


class SearchTwoTowerDeployableModel(TwoTowerModel):
    def init(self, example: ExampleType) -> None:
        # Init lazy layers by going through the forward pass with the training example
        super().init(example)
        # deployable pin tower
        self.candidate_tower_model = self.get_tower_deploy_info(
            example, self.candidate_tower, CANDIDATE_SIG_ITEM_FEATURE_GROUP_NAMES
        )
        # append dummy mutihead weights for torchscript deploy info
        example[MULTI_HEAD_WEIGHTS_FEATURE] = torch.ones(self.batch_size, NUM_HEADS, dtype=torch.int8).to(self.device)
        self.context_tower_model = self.get_tower_deploy_info(
            example, self.context_tower, CONTEXT_FEATURE_GROUP_NAMES, is_multihead=True
        )

    def get_tower_deploy_info(
        self,
        example: ExampleType,
        tower: nn.Module,
        feature_group_names: set(List),
        output_names: List[str] = OUTPUT_NAMES,
        is_multihead: bool = False,
    ) -> nn.Module:
        """
        This function takes in the feature groups and towers to generate the torchscript deploy info
        :param example: Example input batch of tensors
        :param feature_group_names:feature group names
        :param tower: specific tower: candidate or context
        :return: tower model
        """
        if tower is None:
            return None
        tower_model = (
            MultiHeadContextTower(tower)
            if is_multihead
            else nn.Sequential(tower)
        )
        input_names, sample_inputs = self.get_input_info(example, feature_group_names)
        tower_model.DEPLOY_INFO = TorchScriptDeployInfo(
            input_names=input_names,
            output_names=output_names,
            sample_inputs=sample_inputs,
            device=self.device,
            convert_precision=tuple([3] * len(OUTPUT_NAMES)),
        )
        tower_model.torchscript_input_order = input_names
        return tower_model

    def get_input_info(self, example: ExampleType, feature_group_names: set(List)) -> Tuple[List, List]:
        """
        :param example: Example input batch of tensors
        :param feature_group_names: group names
        :return: input names and sample inputs
        """
        input_names = []
        sample_inputs = []
        for key, value in example.items():
            if key.split("/")[0] in feature_group_names:
                input_names.append(key)
                sample_inputs.append(value)
        return input_names, sample_inputs
