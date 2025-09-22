from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import logging
import os
import urllib.parse
from functools import partial
from xml.dom import minidom

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import simplejson as json
import torch
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from interactrank.common.accumulators import AUCAccumulator
from interactrank.common.utils.utils import send_to_device
from interactrank.common.utils.utils import compute_request_level_roc_auc_from_parquet

from interactrank.common.utils.utils import mkdir_p
from interactrank.data.lw_features import FeatureDefinition
from interactrank.configs.base_configs import RELEVANCE_TASK
from interactrank.configs.base_configs import ENGAGEMENT_HEAD
from interactrank.constants.base_constants import ENTITY_ID_FIELD
from interactrank.constants.base_constants import FRESH_PR_AUC_FIELD
from interactrank.constants.base_constants import FRESH_ROC_AUC_FIELD
from interactrank.constants.base_constants import HITS_AT_3
from interactrank.constants.base_constants import IMG_SIG_FIELD
from interactrank.constants.base_constants import LABEL_FIELD
from interactrank.constants.base_constants import PIN_EMBEDDING_FIELD
from interactrank.constants.base_constants import PIN_ID_FIELD
from interactrank.constants.base_constants import PR_AUC_FIELD
from interactrank.constants.base_constants import PR_KEY
from interactrank.constants.base_constants import PREDICTIONS
from interactrank.constants.base_constants import QUERY_EMBEDDING_FIELD
from interactrank.constants.base_constants import QUERY_SEGMENTED_ROC_AUC_FIELD
from interactrank.constants.base_constants import QUERY_STR_FIELD
from interactrank.constants.base_constants import REQ_LEVEL_ROC_AUC
from interactrank.constants.base_constants import REQUEST_ID_FIELD
from interactrank.constants.base_constants import ROC_AUC_FIELD
from interactrank.constants.base_constants import ROC_KEY
from interactrank.configs.base_configs import SEARCH_LW_HEAD_CONFIGS
from interactrank.constants.base_constants import SHOPPING_PR_AUC_FIELD
from interactrank.constants.base_constants import SHOPPING_ROC_AUC_FIELD
from interactrank.constants.base_constants import USER_ID_FIELD
from interactrank.constants.base_constants import WEIGHT_FIELD
from interactrank.feature_consts import REQUEST_ID_FEATURE, USER_ID_FEATURE, SEARCH_QUERY_FEATURE,  IMAGE_SIGNATURE_FEATURE, ITEM_ID_FEATURE, IS_REPIN_FEATURE, IS_LONGCLICK_FEATURE, QUERY_SEGMENT_KEY
from interactrank.common.types import get_label_name_for_training
from interactrank.feature_consts import REPLAY_INPUT,REWARD_VECTOR_KEY,SCORES_KEY

DEFAULT_MIN_NUM_ITEMS_TO_EVAL = 3000
# recall @ k metrics
DEFAULT_RECALL_K = (10, 100, 1000)
DEFAULT_NUM_ENTITY_LIMIT = 1000000
DEFAULT_NUM_USER_LIMIT = 400000
DEFAULT_STRIDE = 256
Z95 = 1.96  # z-score for 95% confidence interval
logger = logging.getLogger(__name__)

PARQUET_DIR = "parquet_dir"
FINAL_SCORES = "final_scores"


def get_schema_tabular(enable_cross_features: bool, all_heads: List[str] = None) -> pa.Schema:
    FIELDS_LR = [
        pa.field(
            USER_ID_FIELD,
            pa.int64(),
            nullable=False,
            metadata={REPLAY_INPUT: USER_ID_FEATURE},
        ),
        pa.field(ENTITY_ID_FIELD, pa.int64()),
        pa.field(
            IMG_SIG_FIELD, pa.string(), metadata={REPLAY_INPUT: ITEM_ID_FEATURE}
        ),
        pa.field(QUERY_STR_FIELD, pa.string()),
        pa.field(PIN_ID_FIELD, pa.int64()),
        pa.field(QUERY_EMBEDDING_FIELD, pa.list_(pa.float32())),
        pa.field(PIN_EMBEDDING_FIELD, pa.list_(pa.float32())),
        pa.field(
            REQUEST_ID_FIELD,
            pa.int64(),
            metadata={REPLAY_INPUT: REQUEST_ID_FEATURE},
        ),
        *[
            pa.field(
                name="_".join([head_name, LABEL_FIELD]),
                type=pa.float32(),
                metadata={REPLAY_INPUT: REWARD_VECTOR_KEY},
            )
            for head_name in all_heads
        ],
        *[
            pa.field(
                name="_".join([head_name, PREDICTIONS]),
                type=pa.float32(),
                metadata={REPLAY_INPUT: SCORES_KEY},
            )
            for head_name in all_heads
        ],
        *[
            pa.field(
                name="_".join([head_name, WEIGHT_FIELD]),
                type=pa.float32(),
            )
            for head_name in all_heads
        ],
    ]

    FIELDS_LWS = [
        *FIELDS_LR,
        pa.field(FINAL_SCORES, pa.float32()),
    ]

    SCHEMA = pa.schema(FIELDS_LWS if enable_cross_features else FIELDS_LR, metadata={"version": "1"})

    return SCHEMA


REQUEST_ID_LABEL = get_label_name_for_training(REQUEST_ID_FEATURE)
USER_ID_LABEL = get_label_name_for_training(USER_ID_FEATURE)
PIN_ID_LABEL = get_label_name_for_training(ITEM_ID_FEATURE)
IMG_SIG_LABEL = get_label_name_for_training(IMAGE_SIGNATURE_FEATURE)
SEARCH_QUERY_LABEL = get_label_name_for_training(SEARCH_QUERY_FEATURE)
REPIN_LABEL = get_label_name_for_training(IS_REPIN_FEATURE)
LONG_CLICK_LABEL = get_label_name_for_training(IS_LONGCLICK_FEATURE)

def get_accumulators(all_heads, enable_cross_features, root_run_dir = None):
    accumulators = {
        ROC_AUC_FIELD: AUCAccumulator(
            predictions_key=f"engagement_{PREDICTIONS}",
            labels_key=f"engagement_{LABEL_FIELD}",
            weights_key=f"engagement_{WEIGHT_FIELD}",
            num_thresholds=2000,
            curve=ROC_KEY,
        ),
        PR_AUC_FIELD: AUCAccumulator(
            predictions_key=f"engagement_{PREDICTIONS}",
            labels_key=f"engagement_{LABEL_FIELD}",
            weights_key=f"engagement_{WEIGHT_FIELD}",
            num_thresholds=2000,
            curve=PR_KEY,
        )
    }
    accumulators.update(
        {
            f"{head}_{metric}": AUCAccumulator(
                predictions_key=f"{head}_{PREDICTIONS}",
                labels_key=f"{head}_{LABEL_FIELD}",
                weights_key=f"{head}_{WEIGHT_FIELD}",
                num_thresholds=2000,
                curve=metric,
            )
            for metric in [ROC_KEY, PR_KEY]
            for head in all_heads
        }
    )
    return accumulators


def compute_all_ranks(
    metric_prefix: str,
    viewer_embeddings: torch.Tensor,
    entity_embeddings: torch.Tensor,
    entity_ids: torch.Tensor,
    user_ids: torch.Tensor,
    image_sigs: np.array,
    labels: torch.Tensor,
    entity_embedding_corpus: torch.Tensor,
    entity_id_corpus: torch.Tensor,
    save_artifacts_fn: Callable[[Union[List[Any], Dict[str, Any]]], None],
    limit: int = DEFAULT_NUM_ENTITY_LIMIT,
    stride: int = DEFAULT_STRIDE,
):
    """
    get ranks for the positive candidates
    given viewer and entity embeddings

    Method Description:
        it computes the recall @ k metric given the negative index corpus.

    logits = torch.where(
            accidental_hits,
            torch.tensor(float("-inf"), device=accidental_hits.device),
            viewer_embeddings_i @ entity_embedding_corpus.T,
        )
    -> This line will make all logits in (viewer_embeddings_i @ entity_embedding_corpus.T)
        which actually corresponds to the positive item to have -inf value.
        so this accidental hit won’t affect final recall calculation.
        this is because in the negative corpus we built entity_embedding_corpus,
        for a given user - item positive pairs, the positive item
        might be actually in the negative corpus

    entity_ranks.append(torch.sum(logits > true_logits.reshape(-1, 1), axis=-1))

    It’s basically computing the rank of the positive items among all the logits
        computed with viewer embedding @ entity_embedding_corpus.
        entity_embedding_corpus is served as a corpus of negative items.
        the torch sum is computed for each row (viewer) and if some of the dot-product of viewer
        embedding and pin embedding is greater than the true logits computed.
        it means some negative items will have higher rank for that viewer compared
        with the corresponding positive items.

    :param metric_prefix: the prefix for the metric
    :param viewer_embeddings: torch.Tensor of shape [B, D]
    :param entity_embeddings: torch.Tensor of shape [B, D]
    :param labels: torch.Tensor of shape [B] whether it is a positive viewer / item pair
    :param entity_embedding_corpus: torch.Tensor of shape [B, D],
        the embedding corpus to evaluate retrieval
    :param entity_id_corpus: torch.Tensor of shape [B, D], the IDs of the entities in the corpus
    :param limit: integer, approximate limit of the number of entity to test
    :param stride: integer, the number of viewers to score in a batch

    :return: the ranks given viewer and positive embedding pairs, rank starting from 0
    """
    example_count = 0
    mask = labels.bool()
    viewer_embeddings = viewer_embeddings[mask][:limit]
    entity_embeddings = entity_embeddings[mask][:limit]
    entity_ids = entity_ids[mask][:limit]
    user_ids = user_ids[mask][:limit]
    logger.info(f"{len(set(user_ids))} unique users for {metric_prefix}")
    true_image_sigs = image_sigs[mask.cpu()][:limit]
    logger.info(f"Entity embedding corpus of size {entity_embedding_corpus.shape[0]} for {metric_prefix}")

    entity_ranks = []
    for i in range(0, viewer_embeddings.shape[0], stride):
        viewer_embeddings_i = viewer_embeddings[i : i + stride]
        num_examples = viewer_embeddings_i.shape[0]
        entity_embeddings_i = entity_embeddings[i : i + stride]
        entity_ids_i = entity_ids[i : i + stride]
        user_ids_i = user_ids[i : i + stride]
        image_sigs_i = true_image_sigs[i : i + stride]

        true_logits = torch.sum(viewer_embeddings_i * entity_embeddings_i, dim=1)
        raw_logits = viewer_embeddings_i @ entity_embedding_corpus.T

        accidental_hits = entity_id_corpus == entity_ids_i.reshape(-1, 1)
        masked_logits = torch.where(
            accidental_hits,
            torch.tensor(float("-inf"), device=accidental_hits.device),
            raw_logits,
        )
        # use >= to avoid the case most entity embs are same
        entity_ranks.append(torch.sum(masked_logits >= true_logits.reshape(-1, 1), axis=-1))
        if save_artifacts_fn and i == 0 and image_sigs_i[0] != b"":
            save_artifacts_fn(
                logits=raw_logits, true_logits=true_logits, user_ids=user_ids_i, true_image_sigs=image_sigs_i
            )
        example_count += num_examples
        if example_count >= limit:
            break

    logger.info(f"{example_count} examples evaluated for {metric_prefix}")
    return torch.cat(entity_ranks)


class TwoTowerLightweightEvaluator:
    """
    Tool for evaluation on two-tower prediction results.
    """

    def __init__(
        self,
        input_dir: str,
        device: torch.device,
        recall_k: Tuple[str, ...] = DEFAULT_RECALL_K,
        entity_limit: int = DEFAULT_NUM_ENTITY_LIMIT,
        user_limit: int = DEFAULT_NUM_USER_LIMIT,
        eval_user_state: bool = False,
        global_bias: np.array = None,
    ):
        """
        :param input_dir: Directory storing predictions files
        :param device: torch.device the device to do evaluation
        :param recall_k: the recall @ k to calculate recall
        :param entity_limit: the number of entity to evaluate
        :param user_limit: the number of user to evaluate
        :param eval_user_state: whether to eval on different user state
        :parma global_bias: np.array of the model bias
        """
        self.device = device
        self.recall_k = recall_k  # define this
        self.entity_limit = entity_limit  # define this
        self.user_limit = user_limit  # (query_limit)
        self.eval_user_state = eval_user_state  # see if we need this?
        self.global_bias = global_bias
        # input_dir can be "" for testing.
        self.dataset = ds.dataset(input_dir, partitioning="hive") if input_dir else None

    def read_data(self):
        # do we need user id here or query id?
        label_field = "engagement_" + LABEL_FIELD
        dataframe = self.dataset.to_table(columns=[USER_ID_FIELD, label_field]).to_pandas()
        b_labels = np.cast[np.bool](dataframe[label_field].to_numpy())
        pos_user_ids = dataframe[USER_ID_FIELD].to_numpy()[b_labels]
        _, unique_indices = np.unique(pos_user_ids, return_index=True)
        # np.unique returns the user_ids in order.  So, deterministically shuffle them.
        rng = np.random.default_rng(seed=0)
        rng.shuffle(unique_indices)
        unique_pos_indices = np.where(b_labels)[0][unique_indices]
        # override the dataframe with rows that have postive labels and change subsequent pin ids, img sig, entity ids, labels etc.
        dataframe = self.dataset.take(indices=unique_pos_indices).to_pandas()
        self.user_ids = dataframe[USER_ID_FIELD].to_numpy()
        self.entity_ids_np = dataframe[ENTITY_ID_FIELD].to_numpy()
        self.image_sigs = dataframe[IMG_SIG_FIELD].to_numpy(dtype="S32")
        self.pin_ids_np = dataframe[PIN_ID_FIELD].to_numpy()
        self.search_queries_np = dataframe[QUERY_STR_FIELD].str.encode("utf-8")
        self.labels = {}
        self.labels[label_field] = dataframe[label_field].to_numpy()
        self.query_embedding_np = np.stack(dataframe[QUERY_EMBEDDING_FIELD])
        self.pin_embedding_np = np.stack(dataframe[PIN_EMBEDDING_FIELD])
        self.search_queries = self.search_queries_np.to_numpy(
            dtype=str("S" + get_array_max_length(self.search_queries_np.to_numpy()))
        )

    def read_entity_corpus(self):
        entity_ids = self.dataset.to_table(columns=[ENTITY_ID_FIELD]).to_pandas()[ENTITY_ID_FIELD].to_numpy()
        _, indices, counts = np.unique(entity_ids, return_index=True, return_counts=True)
        order = np.argsort(-counts)
        indices = indices[order[: self.entity_limit]]
        self.corpus_entity_ids = torch.tensor(entity_ids[indices], device=self.device)
        columns = [PIN_ID_FIELD, PIN_EMBEDDING_FIELD] + (
            [IMG_SIG_FIELD] if IMG_SIG_FIELD in self.dataset.schema.names else []
        )
        dataframe = self.dataset.take(indices=indices, columns=columns).to_pandas()
        self.corpus_pin_ids = torch.tensor(dataframe[PIN_ID_FIELD].to_numpy(), device=self.device)
        # corpus_image_sigs is only used for visualization/debugging.
        self.corpus_image_sigs = dataframe[IMG_SIG_FIELD].to_numpy(dtype="S32") if IMG_SIG_FIELD in dataframe else None
        self.corpus_embeddings = torch.tensor(np.stack(dataframe[PIN_EMBEDDING_FIELD]), device=self.device)

    def compute_ranks(
        self,
        min_num_items_to_eval: int = DEFAULT_MIN_NUM_ITEMS_TO_EVAL,
        run_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        # after read the inference data, compute recall ranks for each user state
        :return: ranks: a map from ('b', 'repin', ...) -> a map from key ('ALL', 'CORE', 'CASUAL') -> eval ranks
        """
        entity_ids = torch.from_numpy(self.entity_ids_np).to(self.device)
        image_sigs = self.image_sigs
        user_ids = torch.from_numpy(self.user_ids).to(self.device)

        all_query_embeddings = torch.from_numpy(self.query_embedding_np).to(self.device)
        all_pin_embeddings = torch.from_numpy(self.pin_embedding_np).to(self.device)
        label_ranks = {}
        for label_field in self.labels:
            logger.info(f"compute rank for label {label_field}")
            labels = torch.from_numpy(self.labels[label_field]).to(self.device)
            all_ranks = compute_all_ranks(
                metric_prefix="ALL",
                viewer_embeddings=all_query_embeddings,
                entity_embeddings=all_pin_embeddings,
                entity_ids=entity_ids,
                user_ids=user_ids,
                image_sigs=image_sigs,
                labels=labels,
                entity_embedding_corpus=self.corpus_embeddings,
                entity_id_corpus=self.corpus_entity_ids,
                limit=self.user_limit,
            )
            label_ranks[label_field] = {"ALL": all_ranks}
            return label_ranks

    def compute_recall_at_k_metric(self, entity_ranks_dict: Dict[str, torch.Tensor]) -> Dict[str, tuple]:
        """
        :param entity_ranks_dict: A map of [Str, 1D torch.Tensor of all entity ranks]
        return recall at k results as well as p95 error for each rank
        """
        recall_tensor = torch.from_numpy(np.array([self.recall_k])).to(self.device)
        results = {}
        for user_state in entity_ranks_dict:
            user_entity_ranks = entity_ranks_dict[user_state]
            user_count = user_entity_ranks.shape[0]
            hits_at_k = torch.sum(user_entity_ranks.reshape(-1, 1) < recall_tensor, axis=0)
            p_recall = hits_at_k / user_count
            std_err = torch.sqrt(p_recall * (1 - p_recall) / user_count)
            p95_err = Z95 * std_err
            results[user_state] = (p_recall, p95_err)
        return results

def two_tower_lightweight_post_inference_fn(
    results,
    run_dir: str,
    iteration: int,
    metric_prefix: str,
    world_size: int,
    device: str,
    model: nn.Module,
    enable_compute_recall: bool,
    enable_estimator_rewards: bool,
    summary_writer: Optional[SummaryWriter] = None,
    **kwargs,
) -> None:
    """
    This function reads the data returned by `two_tower_lightweight_inference_fn` and then compute metrics on them.
    :param results: dict of metrics computed
    :param run_dir: path to store model stats
    :param iteration: number of iterations
    :param metric_prefix: used to generate stats path
    :param world_size: world size
    :param device: device type
    :param model: Model
    :param enable_compute_recall: compute recall metrics along with visualization
    :return: None; log the computed metrics
    """
    metrics_names = set(
        [
            ROC_AUC_FIELD,
            SHOPPING_ROC_AUC_FIELD,
            FRESH_ROC_AUC_FIELD,
            QUERY_SEGMENTED_ROC_AUC_FIELD,
            PR_AUC_FIELD,
            SHOPPING_PR_AUC_FIELD,
            FRESH_PR_AUC_FIELD,
            REQ_LEVEL_ROC_AUC,
            HITS_AT_3,
            *[f"{head}_{metric}" for metric in [ROC_KEY, PR_KEY] for head in model.label_extractor.engagement_heads],
            *[f"mm_{metric}" for metric in [ROC_KEY, PR_KEY, SHOPPING_ROC_AUC_FIELD, SHOPPING_PR_AUC_FIELD]],
        ]
    )
    model_stats_dir = os.path.join(run_dir, "model_stats", "iteration_" + str(iteration), metric_prefix)
    mkdir_p(model_stats_dir)

    # add recall metrics
    if enable_compute_recall and PARQUET_DIR in results:
        eval_dirs = [None] * 1
        eval_dirs[0] = results[PARQUET_DIR]
        global_bias = model.global_bias.detach().cpu().numpy()
        evaluator = TwoTowerLightweightEvaluator(device=device, input_dir=eval_dirs[0], global_bias=global_bias)
        evaluator.read_data()
        evaluator.read_entity_corpus()
        recall_metrics = two_tower_recall_eval(evaluator=evaluator, run_dir=model_stats_dir)
        for label_field, user_state_metrics in recall_metrics.items():
            for user_state, recall_dict in user_state_metrics.items():
                for k in recall_dict:
                    name = os.path.join(metric_prefix, "recall_eval", f"{label_field}", user_state + "_" + str(k))
                    summary_writer.add_scalar(name, recall_dict[k][0], global_step=iteration * world_size)
    # add request level metrics, hits@3 metric
    if PARQUET_DIR in results:
        results[REQ_LEVEL_ROC_AUC] = compute_request_level_roc_auc_from_parquet(
            parquet_dir=results[PARQUET_DIR],
            device=device,
            request_id_field=REQUEST_ID_FIELD,
            prediciton_field=f"{ENGAGEMENT_HEAD}_{PREDICTIONS}",
            label_field=f"{ENGAGEMENT_HEAD}_{LABEL_FIELD}",
        )

    # print the metrics
    print("Metrics!!!")
    for metric_name, metric_value in results.items():
        print(f"{os.path.join(metric_prefix, 'eval', metric_name)}: {metric_value}")

    if summary_writer is not None:
        for metric_name, metric_value in results.items():
            if metric_name == QUERY_SEGMENTED_ROC_AUC_FIELD:
                for query_segment, roc_auc in metric_value.items():
                    summary_writer.add_scalar(os.path.join(metric_prefix, "eval", QUERY_SEGMENTED_ROC_AUC_FIELD, query_segment),roc_auc.item(),iteration * world_size)
            elif metric_name in metrics_names and metric_name != PARQUET_DIR:
                summary_writer.add_scalar(os.path.join(metric_prefix, "eval", metric_name), metric_value.item(), iteration * world_size)


def get_array_max_length(input_array):
    "This function returns the length of string with max length, this is used to define search query numpy array dtype"
    length_checker = np.vectorize(len)
    arr_len = length_checker(input_array)
    return str(arr_len.max())


def two_tower_tabular_lightweight_inference_fn(
    example: Dict[str, Tensor],
    model: nn.Module,
    metamodel_weights: List[float],
    saved_results_cols: Optional[List[FeatureDefinition]] = None,
    task: str = "",
) -> Dict[str, Tensor]:
    """
    This functions reads the eval data in batches and then using then compute dot product between query and pin
    tower and then concatenate it with cross features to compute final scores for the tabular ML data
    :param example: evaluation data
    :param model: trained model
    :param saved_results_cols: if set, will save additional cols together with inference results.
    :param task: task name
    :return: dict of scores, labels, weights, predictions
    """
    original_mode = model.training
    model.eval()
    example = send_to_device(example, model.device, ignore_error=True)
    final_scores, context_tower_embeddings, candidate_tower_output = model.eval_forward(example)

    (labels, weights), _ = model.label_extractor(example)

    if final_scores is not None:
        predictions = torch.sigmoid(final_scores)
        final_scores = predictions[:, 0]
    else:
        if context_tower_embeddings is None:
            predictions = candidate_tower_output
        else:
            predictions = torch.sigmoid(
                torch.sum(context_tower_embeddings * candidate_tower_output.unsqueeze(1), dim=2) + model.global_bias
            )

    model.train(mode=original_mode)
    with torch.no_grad():
        all_heads = [config.name.value for config in SEARCH_LW_HEAD_CONFIGS]
        predictions = dict(zip(all_heads, predictions.T))
        results = {}
        for idx, head in enumerate(all_heads):
            results[f"{head}_{PREDICTIONS}"] = torch.detach(predictions[head])
            results[f"{head}_{LABEL_FIELD}"] = torch.detach(labels[:, idx])
            results[f"{head}_{WEIGHT_FIELD}"] = torch.detach(weights[:, idx])

        metamodel_score = metamodel_score_computer(predictions, metamodel_weights=metamodel_weights)
        results["metamodel_predictions"] = torch.detach(metamodel_score)

        # relevance data doesnt have request Ids
        if task == RELEVANCE_TASK:
            request_ids = torch.zeros_like(labels[:, 0])
            user_ids = torch.zeros_like(labels[:, 0])
            pin_ids = torch.zeros_like(labels[:, 0])
            img_sigs = torch.zeros_like(labels[:, 0])
            entity_ids = torch.zeros_like(labels[:, 0])
            # search_queries = torch.zeros_like(labels[:, 0])
        else:
            request_ids = torch.tensor([int(req) for req in example[REQUEST_ID_LABEL]])
            user_ids = torch.tensor([int(userid) for userid in example[USER_ID_LABEL]])
            pin_ids = torch.tensor([int(pinid) for pinid in example[PIN_ID_LABEL]])
            # byte_array = example[IMG_SIG_LABEL].cpu().numpy().tobytes()
            img_sigs = example[IMG_SIG_LABEL].cpu().numpy()
            entity_ids = torch.tensor([int(pinid) for pinid in example[PIN_ID_LABEL]])
          #  arr_len_max = get_array_max_length(example[SEARCH_QUERY_LABEL])
           # search_queries = np.frombuffer(example[SEARCH_QUERY_LABEL].data, dtype=str("S" + arr_len_max))

        results.update(
            {
                FINAL_SCORES: final_scores,
                REQUEST_ID_FIELD: request_ids,
                USER_ID_FIELD: user_ids,
                PIN_ID_FIELD: pin_ids,
                IMG_SIG_FIELD: img_sigs,
               # QUERY_STR_FIELD: search_queries,
                ENTITY_ID_FIELD: entity_ids,
                QUERY_EMBEDDING_FIELD: context_tower_embeddings[:, 0, :],
                PIN_EMBEDDING_FIELD: candidate_tower_output,
              #  IS_TWP_FIELD: example[IS_TRUSTWORTHY_PRODUCT_LABEL],
            }
        )

    return results


def metamodel_score_computer(predictions: Dict, metamodel_weights: List[float]) -> Tensor:
    # prediction keys: 'engagement', 'repin', 'longclick'
    # Each value is of shape (batch_size,)
    for head_config in SEARCH_LW_HEAD_CONFIGS:
        assert head_config.name.value in predictions, f"{head_config.name.value} not in predictions"
    assert len(SEARCH_LW_HEAD_CONFIGS) == len(metamodel_weights), (
        f"There are {len(SEARCH_LW_HEAD_CONFIGS)} heads but {len(metamodel_weights)} head weights"
    )

    metamodel_score = torch.zeros_like(list(predictions.values())[0])
    for metamodel_weight, head_config in zip(metamodel_weights, SEARCH_LW_HEAD_CONFIGS):
        metamodel_score += metamodel_weight * predictions[head_config.name.value]
    return metamodel_score


def two_tower_recall_eval(
    evaluator: TwoTowerLightweightEvaluator,
    run_dir: Optional[str] = None,
    recall_k: Tuple[str, ...] = DEFAULT_RECALL_K,
) -> Dict[str, Dict[str, tuple]]:
    """
    :param evaluator: TwoTowerEvaluator to read the eval file and generate eval examples
    :param run_dir: (Optional) Directory to store the model stats
    :param recall_k: tuple of recall at k to test

    :return: Tuple of aggregated metrics per label and per user metrics
    """
    # get recall @ k results
    # entity_ranks_dict Dict[label, Dict[user_state, rank]]
    entity_ranks_dict = evaluator.compute_ranks(min_num_items_to_eval=DEFAULT_MIN_NUM_ITEMS_TO_EVAL, run_dir=run_dir)

    all_eval_stats = {}
    label_field = f"{ENGAGEMENT_HEAD}_{LABEL_FIELD}"
    entity_ranks_dict_label = entity_ranks_dict[label_field]
    results = evaluator.compute_recall_at_k_metric(entity_ranks_dict=entity_ranks_dict_label)
    eval_stats = {}
    for metric_prefix in results:
        recall_result = results[metric_prefix][0]
        p95_err = results[metric_prefix][1]
        recall_result = recall_result.cpu().numpy().tolist()
        p95_err = p95_err.cpu().numpy().tolist()
        eval_stats[metric_prefix] = {}
        for k_idx, k in enumerate(recall_k):
            eval_stats[metric_prefix][k] = (recall_result[k_idx], p95_err[k_idx])
            print(
                f"label: {label_field} metric_prefix: {metric_prefix} \t recalls at {k}: {recall_result[k_idx]} \t p95_err: {p95_err[k_idx]}"
            )
    if run_dir:
        with open(os.path.join(run_dir, f"{label_field}_model_stats_file.json"), "w") as fptr:
            fptr.write(json.dumps(eval_stats))
    all_eval_stats[label_field] = eval_stats

    return all_eval_stats
