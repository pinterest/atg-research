from __future__ import annotations

from typing import Collection
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import preprocessing
from torch import nn
from interactrank.common.utils.utils import send_to_device
from interactrank.label_extractor import TabularLightweightLabelExtractor
from interactrank.configs.base_configs import Metric
from interactrank.common.utils.model_utils import convert_img_sig_tensor_to_long
from interactrank.common.utils.model_utils import maybe_dedup_negatives
from interactrank.common.utils.model_utils import CountMinSketch
from interactrank.common.utils.model_utils import InBatchNegativeLossMultihead
from interactrank.common.utils.metrics import BinnedAUC
from interactrank.common.utils.model_utils import AllGatherWithGrad

from interactrank.feature_consts import SEARCH_QUERY_FEATURE, IMAGE_SIGNATURE_FEATURE
from interactrank.common.types import EntityType
from interactrank.eval import get_label_name_for_training
from interactrank.configs.base_configs import NUM_HEADS
from interactrank.configs.base_configs import DOT_PRODUCT_FEATURE_FIELD
from interactrank.configs.base_configs import LABEL_FIELD
from interactrank.configs.base_configs import LABEL_TYPE
from interactrank.configs.base_configs import PIN
from interactrank.configs.base_configs import PR_AUC
from interactrank.configs.base_configs import PREDICTIONS_FIELD
from interactrank.configs.base_configs import QUERY
from interactrank.configs.base_configs import ROC_AUC
from interactrank.configs.base_configs import SEARCHSAGE_VERSION_TO_FEATURE
from interactrank.configs.base_configs import SOFTMAX_LOSS
from interactrank.configs.base_configs import TOTAL_LOSS
from interactrank.configs.base_configs import WEIGHT_FIELD
from interactrank.common.utils.utils import ExampleType

COUNT_MIN_SKETCH_DEPTH = 1
COUNT_MIN_SKETCH_WIDTH = 1 << 28
COUNT_MIN_SKETCH_SEED = 12451

np.seterr(divide="ignore", invalid="ignore")
FIELDS_TO_SKIP = [LABEL_FIELD, WEIGHT_FIELD, LABEL_TYPE]
IMAGE_SIGNATURE = get_label_name_for_training(IMAGE_SIGNATURE_FEATURE)
SEARCH_QUERY = get_label_name_for_training(SEARCH_QUERY_FEATURE)


def binary_focal_loss(predictions, labels, weights, gamma=2):
    labels = labels.float()
    batch_logloss = F.binary_cross_entropy_with_logits(predictions, labels, reduction="none")
    pt = torch.exp(-batch_logloss)
    batch_focal_loss = batch_logloss * ((1 - pt) ** gamma)
    batch_focal_loss = batch_focal_loss * weights
    return batch_focal_loss

class TwoTowerModel(nn.Module):
    """
    This is a generic two-tower model code
    """

    def __init__(
        self,
        label_map: Collection[str],
        label_extractor: TabularLightweightLabelExtractor,
        device: torch.device,
        context_tower: nn.Module,
        candidate_tower: nn.Module,
        cross_layer: nn.Module,
        use_in_batch_negatives: bool,
        in_batch_neg_rel_weight: int,
        correct_sample_probability: bool,
        searchsage_eval_only_with_version: str,
        logged_relevance_loss_weight: int,
        batch_size: int,
        loss_function: str = "bce",
        upweight_tprc_factor: float = 1.0,
        in_batch_neg_mask_on_queries: bool = False,
    ):
        """
        :param label_map: label map
        :param label_extractor: used for tabularML dataset
        :param device: The device to use.
        :param context_tower: Tower defined for context data.
        :param candidate_tower: Tower defined for candidate data.
        :param cross_layer: Layer that takes in cross feature data & dot product.
        :param feature_map: Feature Map
        :param use_in_batch_negatives: Whether we want to compute in batch negative loss,
        :param in_batch_neg_rel_weight: Weight applied to the neg loss.
        :param correct_sample_probability: when set to true, it will apply item counter to correct the sample
         probability of items in the batch
        """
        super().__init__()

        self.device = device
        self.label_map = label_map
        self.label_extractor = label_extractor
        self.register_parameter("global_bias", nn.Parameter(torch.zeros(NUM_HEADS)))
        self.context_tower = context_tower
        self.candidate_tower = candidate_tower
        self.cross_layer = cross_layer
        self.accumulated_pr_auc = BinnedAUC(num_thresholds=2000, curve="PR", accumulated=False)
        self.accumulated_roc_auc = BinnedAUC(num_thresholds=2000, curve="ROC", accumulated=False)
        self.fields_to_skip = FIELDS_TO_SKIP
        self.use_in_batch_negatives = use_in_batch_negatives
        self.in_batch_neg_rel_weight = in_batch_neg_rel_weight
        self.correct_sample_probability = correct_sample_probability
        self.in_batch_neg_mask_on_queries = in_batch_neg_mask_on_queries
        if self.in_batch_neg_mask_on_queries:
            self.query_encoder = preprocessing.LabelEncoder()
        self.upweight_tprc_factor = upweight_tprc_factor
        self.searchsage_eval_only_with_version = searchsage_eval_only_with_version
        self.logged_relevance_loss_weight = logged_relevance_loss_weight
        if self.use_in_batch_negatives:
            self.all_gather_negatives = False
            if self.correct_sample_probability:
                # Define an item counter using CountMinSketch.
                self.item_counter = CountMinSketch(
                    w=COUNT_MIN_SKETCH_WIDTH,
                    d=COUNT_MIN_SKETCH_DEPTH,
                    seed=COUNT_MIN_SKETCH_SEED,
                    synchronize_counts=not self.all_gather_negatives,
                )
            else:
                self.item_counter = None  # Disable item_counter for now
            self.in_batch_negative_loss = InBatchNegativeLossMultihead(
                item_counter=self.item_counter,
            )
        self.loss_function = loss_function
        self.embedding_dim = context_tower.embedding_dim
        self.engagement_normalized_heads_weights = self.label_extractor.engagement_normalized_heads_weights
        self.batch_size = batch_size
        self.to(self.device)

    def init(self, example: ExampleType) -> None:
        # Init lazy layers by going through the forward pass with the training example
        model_mode = self.training
        self.train(mode=False)
        self.forward(example)
        self.train(mode=model_mode)

    def get_batch_log_loss(
        self,
        final_scores: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
        context_tower_embeddings: torch.Tensor,
        candidate_tower_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param final_scores: tensor size  [BatchSize, 1]
        :param labels: labels used for training [BatchSize]
        :param weights: label weights used for training [BatchSize]
        :param context_tower_embeddings: Batch left tower embeddings [BatchSize, EmbeddingSize]
        :param candidate_tower_output: Batch right tower embeddings [BatchSize, EmbeddingSize]
        / Scores in case of static rank
        :return: a tuple of torch tensor represents the batch_logloss and predictions
        """
        if final_scores is not None:
            predictions = final_scores
        else:
            if context_tower_embeddings is None:
                predictions = candidate_tower_output
            else:
                predictions = (
                    torch.sum(context_tower_embeddings * candidate_tower_output.unsqueeze(1), dim=2) + self.global_bias
                )

        if self.loss_function == "bce":
            batch_loss_sum = F.binary_cross_entropy_with_logits(predictions, labels.float(), weights, reduction="none")
        elif self.loss_function == "focal":
            batch_loss_sum = binary_focal_loss(predictions, labels, weights, gamma=2)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")
        return batch_loss_sum, predictions.sigmoid()

    def get_summary_metrics(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Create the dictionary of summary metrics given the results from the forward pass
        :param inputs: dict containing data passed in loss_metrics
        :return dict of summary metrics
        """
        metrics = {}
        with torch.no_grad():
            labels = inputs[LABEL_FIELD]
            weights = inputs[WEIGHT_FIELD]
            predictions = inputs[PREDICTIONS_FIELD]

            metrics[TOTAL_LOSS] = inputs[TOTAL_LOSS].item()
            if self.use_in_batch_negatives:
                metrics[SOFTMAX_LOSS] = inputs[SOFTMAX_LOSS].item()
            metrics[PR_AUC] = self.accumulated_pr_auc(predictions, labels, weights=weights).item()
            metrics[ROC_AUC] = self.accumulated_roc_auc(predictions, labels, weights=weights).item()
            if self.cross_layer:
                metrics.update(self.cross_layer.get_summary_metrics(inputs))

            # label statistics
            binary_labels = labels[:, 0]
            metrics[Metric.LABEL_POS_SUM.name] = binary_labels.sum().item()
            metrics[Metric.LABEL_NEG_SUM.name] = (1 - binary_labels).sum().item()
            metrics[Metric.LABEL_POS_RATE.name] = metrics[Metric.LABEL_POS_SUM.name] / binary_labels.size(0)
            metrics[Metric.WEIGHT_SUM.name] = weights.sum().item()
            metrics[Metric.POS_WEIGHT_SUM.name] = torch.sum(weights[binary_labels == 1]).item()

        return metrics

    def is_cross_feature(self, batch_key) -> bool:
        """
        This function helps return features that maps to cross features entity type
        :param batch_key: key from a batch of examples
        :return: boolean value whether
        """
        for entity in self.cross_layer.get_cross_feature_keys():
            if batch_key.startswith(entity):
                return True
        return False

    def get_final_scores(self, batch: ExampleType, dot_products: torch.Tensor) -> torch.Tensor:
        """
        Takes input the cross features from the batch and dot product value btw context & candidate tower, to
        return the final embedding generated from the same.
        :param batch:The input batch of tensors.
        :param dot_product: between context and candidate embeddings.
        :return: final_scores
        """
        # only return cross features here
        cross_features = {k: v for k, v in batch.items() if self.is_cross_feature(k)}
        formatted_data_new = cross_features
        formatted_data_new[DOT_PRODUCT_FEATURE_FIELD] = dot_products
        final_scores = self.cross_layer(formatted_data_new)
        return final_scores

    def convert_inf(self, input):
        """
        This function replaces inf and nan values with zero
        :param input: Tensor array
        :return: Tensor array data
        """
        input[input == float("Inf")] = 0
        input = torch.nan_to_num(input)
        return input.data

    def _forward(self, batch: ExampleType) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The commonly used forward method for training and eval.
        :param batch:The input batch of tensors.
        :return:final scores, context and candidate embeddings
        """
        if self.cross_layer is not None:
            features = {k: v for k, v in batch.items() if k not in self.fields_to_skip and not self.is_cross_feature(k)}
            formatted_data = features
            context_tower_embeddings = self.context_tower(formatted_data)
            # reshape to [BatchSize, NumHeads, EmbeddingDim]
            context_tower_embeddings = context_tower_embeddings.reshape([-1, NUM_HEADS, self.embedding_dim])
            candidate_tower_embeddings = self.candidate_tower(formatted_data)
            dot_products = torch.sum(context_tower_embeddings * candidate_tower_embeddings.unsqueeze(1), dim=2)
            return (
                self.get_final_scores(batch, dot_products),
                context_tower_embeddings,
                candidate_tower_embeddings,
            )

    def _forward_ss(
        self, batch: ExampleType, searchsage_eval_only_with_version: str
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward method to evaluate searchsage embeddings recall
        This function sets context and candidate towers to searchsage version
        embeddings, which is used to calculate rank and recall later.
        :param batch:The input batch of tensors.
        :return:final scores, context and candidate embeddings
        """

        features = {k: v for k, v in batch.items() if k not in self.fields_to_skip}
        formatted_data = features

        searchsage_query_embedding = SEARCHSAGE_VERSION_TO_FEATURE[searchsage_eval_only_with_version][QUERY]
        searchsage_pin_embedding = SEARCHSAGE_VERSION_TO_FEATURE[searchsage_eval_only_with_version][PIN]

        context_tower_embeddings = formatted_data[searchsage_query_embedding]
        candidate_tower_embeddings = formatted_data[searchsage_pin_embedding]

        context_tower_embeddings = context_tower_embeddings.type("torch.FloatTensor").to(self.device)
        candidate_tower_embeddings = candidate_tower_embeddings.type("torch.FloatTensor").to(self.device)

        return (
            None,
            context_tower_embeddings,
            candidate_tower_embeddings,
        )

    def forward(self, batch: ExampleType) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        The forward pass for training.
        :param batch: The input batch of tensors.
        :return: A tuple of loss and summary metrics dictionaries.
        """
        batch = send_to_device(batch, self.device, ignore_error=True)
        (labels, weights), (logged_relevance_labels, logged_relevance_weights) = self.label_extractor.forward(
            batch, upweight_tprc_factor=self.upweight_tprc_factor if self.training else 1.0
        )
        loss_metrics = {}
        final_scores, context_tower_embeddings, candidate_tower_output = self._forward(batch)
        # add / update metric with log loss
        batch_logloss, predictions = self.get_batch_log_loss(
            final_scores=final_scores,
            labels=labels,
            weights=weights,
            context_tower_embeddings=context_tower_embeddings,
            candidate_tower_output=candidate_tower_output,
        )
         # compute in batch negative loss if 'use_in_batch_negatives' is set to True
        if self.use_in_batch_negatives:
            in_batch_negative_loss = self.compute_in_batch_negative_loss(
                batch, candidate_tower_output, context_tower_embeddings, labels, weights
            )
        else:
            in_batch_negative_loss = 0
        head_losses = torch.mean(batch_logloss, dim=0)
        batch_logloss = torch.sum(
            head_losses * torch.tensor(self.engagement_normalized_heads_weights, device=self.device)
        )

        loss_metrics[SOFTMAX_LOSS] = in_batch_negative_loss
        loss_metrics[LABEL_FIELD] = labels
        loss_metrics[WEIGHT_FIELD] = weights
        loss_metrics[PREDICTIONS_FIELD] = predictions
        # this loss will be used for final calculation
        loss_metrics[TOTAL_LOSS] = batch_logloss + (self.in_batch_neg_rel_weight * in_batch_negative_loss)
        print("Loss metrics: ", loss_metrics[SOFTMAX_LOSS])
        return loss_metrics

    def eval_forward(self, batch: ExampleType) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass for eval.
        :param batch: The input batch of tensors.
        :return: A tuple of context and candidate embeddings.
        """
        batch = send_to_device(batch, self.device, ignore_error=True)
        if self.searchsage_eval_only_with_version is not None:
            return self._forward_ss(batch, self.searchsage_eval_only_with_version)
        else:
            return self._forward(batch)

    def compute_in_batch_negative_loss(
        self,
        batch: ExampleType,
        candidate_tower_output: torch.Tensor,
        context_tower_embeddings: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ):
        """
        This function computes the in batch negative loss for an input batch.
        :param batch: The input batch of tensors.
        :param candidate_tower_output: Batch right tower embeddings [BatchSize, EmbeddingSize]
        :param context_tower_embeddings: Batch left tower embeddings [BatchSize, EmbeddingSize]
        :param labels: labels used for training [BatchSize]
        :param weights: label weights used for training [BatchSize]
        :return: In batch negative loss.
        """

        if self.in_batch_neg_mask_on_queries:
            np_queries = batch[SEARCH_QUERY]
            query_ids = self.query_encoder.fit_transform(np_queries)
            query_ids = torch.from_numpy(query_ids).to(self.device)

            if self.all_gather_negatives:
                negative_query_ids = torch.cat(AllGatherWithGrad.apply(query_ids), dim=0)
                negative_embeddings = torch.cat(AllGatherWithGrad.apply(context_tower_embeddings), dim=0)
            else:
                negative_query_ids = query_ids
                negative_embeddings = context_tower_embeddings

            unique_neg_items, unique_neg_embeddings = maybe_dedup_negatives(
                negative_embeddings=negative_embeddings, negative_items=negative_query_ids, dedup_method="average"
            )
            if self.item_counter is not None:
                self.item_counter.update(unique_neg_items)
            in_batch_negative_loss, _, _ = self.in_batch_negative_loss(
                viewer_embeddings=candidate_tower_output,
                entity_embeddings=context_tower_embeddings,
                true_label_mask=labels.bool(),
                weights=torch.ones_like(weights),
                items=query_ids,
                negative_items=negative_query_ids,
                unique_neg_embeddings=unique_neg_embeddings,
                unique_neg_items=unique_neg_items,
            )
            in_batch_negative_loss /= query_ids.numel()
        else:
            np_sigs = batch[f"{EntityType.LABEL.value}/image_sig"]
            raw_sigs = torch.from_numpy(np_sigs.cpu().numpy()).reshape(np_sigs.shape[0], -1).to(self.device)
            sigs = convert_img_sig_tensor_to_long(raw_sigs)

            if self.all_gather_negatives:
                negative_sigs = torch.cat(AllGatherWithGrad.apply(sigs), dim=0)
                negative_embeddings = torch.cat(AllGatherWithGrad.apply(candidate_tower_output), dim=0)
            else:
                negative_sigs = sigs
                negative_embeddings = candidate_tower_output

            unique_neg_items, unique_neg_embeddings = maybe_dedup_negatives(
                negative_embeddings=negative_embeddings, negative_items=negative_sigs, dedup_method="average"
            )
            if self.item_counter is not None:
                self.item_counter.update(unique_neg_items)
            in_batch_negative_loss, _, _ = self.in_batch_negative_loss(
                viewer_embeddings=context_tower_embeddings,
                entity_embeddings=candidate_tower_output,
                true_label_mask=labels.bool().T,
                weights=torch.ones_like(weights).T,
                items=sigs,
                negative_items=negative_sigs,
                unique_neg_embeddings=unique_neg_embeddings,
                unique_neg_items=unique_neg_items,
            )
            in_batch_negative_loss /= sigs.numel()
        return in_batch_negative_loss
