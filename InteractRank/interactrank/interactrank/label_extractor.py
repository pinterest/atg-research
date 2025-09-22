from __future__ import annotations

from typing import List
from typing import Tuple
from torch import nn

import logging
import copy
from collections import OrderedDict

import torch
from interactrank.common.utils.utils import ExampleType
from interactrank.configs.base_configs import SearchHeadNames
from interactrank.common.types import SearchLabels
from interactrank.configs.bundle import DEFAULT_TABULAR_LONG_CLICK_WEIGHT_LWS
from interactrank.configs.base_configs import SEARCH_LW_HEAD_CONFIGS

logger = logging.getLogger(__name__)

RELEVANCE_HARD_LABEL_THRESHOLD = 0.7
IS_TRUSTWORTHY_PRODUCT_LABEL = SearchLabels.IS_TRUSTWORTHY
LOGGED_RELEVANCE_SCORE_LABEL = SearchLabels.LOGGED_RELEVANCE_SCORE


class TabularLightweightLabelExtractor(nn.Module):

    def __init__(self, device, app_config, **kwargs):
        self.device = device
        # update label map for LWS
        self.update_engagement_label_map_lws()
        # update label map for learned retrieval
        self.app_config = app_config
        super().__init__(**kwargs)
        self.engagement_multi_heads_labels = [
            config.label.value for config in SEARCH_LW_HEAD_CONFIGS if config.name != SearchHeadNames.ENGAGEMENT
        ]
        self.engagement_heads = [config.name.value for config in SEARCH_LW_HEAD_CONFIGS]
        self.engagement_heads_weights = [config.label_weight for config in SEARCH_LW_HEAD_CONFIGS]
        self.engagement_normalized_heads_weights = [
            head_weight / sum(self.engagement_heads_weights) for head_weight in self.engagement_heads_weights
        ]
        self.PRIORITY_ENGAGEMENT_LABEL_TO_WEIGHT: OrderedDict[str, float] = OrderedDict(
            [
                (SearchLabels.LONG_CLICK.value, 20.0),
                (SearchLabels.LONG_CLICK_5S.value, 20.0),
                (SearchLabels.SAVE_TO_DEVICE.value, 7.0),
                (SearchLabels.REPIN.value, 7.0),
               # (SearchLabels.SCREENSHOT.value, 5.0),
               # (SearchLabels.TW_DR_CLICK_5S.value, 20.0),
                (SearchLabels.SHORT_CLICK_5S.value, 20.0),
               # (SearchLabels.SHORT_CLICK_10S.value, 20.0),
                (SearchLabels.CLICK.value, 1.0),
               # (SearchLabels.FULL_SCREEN_VIEW_10S.value, 2.0),
                #(SearchLabels.VIDEO_CLOSEUP_10S.value, 2.0),
                #(SearchLabels.CLOSEUP_10S.value, 2.0),
                (SearchLabels.CLOSEUP.value, 0.5),
                # Generally keep impression last so if a pin is impression-only, it is given this weight
                (SearchLabels.IMPRESSION.value, 5.0),
            ]
        )
        self.engagement_label_to_weight = copy.deepcopy(self.PRIORITY_ENGAGEMENT_LABEL_TO_WEIGHT)
        self.register_buffer(
            "action_weights",
            torch.tensor([[weight] for _, weight in self.engagement_label_to_weight.items()], device=self.device),
            persistent=False,
        )

    def is_positive_engagement(self, example: ExampleType, batch_size: int) -> torch.Tensor:
        """
        Args:
            example: tensor dict mapping feature name to a Tensor with batch size B
            batch_size: size of the current batch being examined
        Returns: B-length 1D Tensor that is True for entries where there is any positive engagement and False otherwise
        """
        # For W_p positive engagement labels, this produces a W_p x B tensor that is True if engagement W_p occurred
        positive_label_match = torch.stack(
            [
                (
                    example[positive_label] > 0
                    if positive_label in example
                    else torch.zeros(batch_size, device=self.device)
                )
                for positive_label in self.POSITIVE_ENGAGEMENT_LABELS
            ]
        )
        # B-length 1D tensor that is True if there is any positive engagement
        any_positive_label_match = torch.max(positive_label_match, dim=0).values
        # For W_n negative engagement labels, this produces a W_n x B tensor that is True if engagement W_p occurred
        negative_label_match = torch.stack(
            [
                (
                    example[negative_label] > 0
                    if negative_label in example
                    else torch.zeros(batch_size, device=self.device)
                )
                for negative_label in self.NEGATIVE_ENGAGEMENT_LABELS
            ]
        )
        # For W_p positive multihead engagement labels,
        # this produces a W_p x B tensor that is True
        # if engagement W_p occurred
        multihead_label_match = torch.stack(
            [
                example[label] > 0 if label in example else torch.zeros(batch_size, device=self.device)
                for label in self.engagement_multi_heads_labels
            ]
        )
        # B-length 1D tensor that is True if there is any negative engagement
        any_negative_label_match = torch.max(negative_label_match, dim=0).values
        # If a clear negative engagement occurs, treat the data point as negative even if it was previously marked
        # positive. Return True for cases that are positive only.
        b_head_labels = torch.logical_and(any_positive_label_match, torch.logical_not(any_negative_label_match))
        # append multihead labels
        engagement_labels = torch.cat([b_head_labels.reshape(1, batch_size), multihead_label_match], dim=0)
        return engagement_labels

    def get_engagement_weights(
        self, example: ExampleType, batch_size: int, upweight_tprc_factor: float = 1.0
    ) -> torch.Tensor:
        """
        Args:
            example: tensor dict mapping feature name to a Tensor with batch size B
            batch_size: size of the current batch being examined
        Returns: B-length 1D Tensor of float weights for each engagement training data point in the batch
        """
        # For W weights in engagement_label_to_weight, produce a W x B tensor describing whether each
        # label is met for the given datapoint. Note that because engagement_label_to_weight is ordered,
        # this is also ordered by decreasing priority (row 0 is highest-priority engagement)
        label_match = torch.stack(
            [
                example[label_name] > 0 if label_name in example else torch.zeros(batch_size, device=self.device)
                for label_name, _ in self.engagement_label_to_weight.items()
            ]
        )

        # For each column (data point), find the first matching engagement (row) using .indices, giving a B-dim tensor
        # where each entry is the lowest-matching index in engagement_label_to_weight.
        # NOTE: In the rare case where there is no matching engagement (including impressions), this will give index 0,
        # which is handled below.
        lowest_matching_indices = torch.max(label_match, dim=0).indices
        # Look up each of the matching indices from action_weights to get preliminary weights
        weight_tensor = self.action_weights.T.squeeze()[lowest_matching_indices]
        # To avoid edge cases where .indices gives 0 for rows with zero matching labels, override these potential
        # edge cases' weights to 1.
        no_matching_labels_mask = torch.sum(label_match, dim=0) == 0
        weight_tensor[torch.where(no_matching_labels_mask)] = 1.0
        # Repeat the weight_tensor to match the shape of engagement heads' labels
        weight_tensor = weight_tensor.reshape((-1, 1)).repeat(1, len(self.engagement_heads))
        return weight_tensor.T

    def forward(
        self, example: ExampleType, upweight_tprc_factor: float = 1.0
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get label and weight tensors for both LW task (similar to engagement data for full scoring).
        Args:
            example: tensor dict mapping feature name to a Tensor with batch size B
        Returns: A tuple of two tuples, in this order:
            Tuple 1 -
                1) A Bx1 label tensor containing 0/1 prediction labels for engagement tasks
                2) A Bx1 weight tensor containing weights for engagement task. The task that
                is not currently being optimized for has its weight column zeroed out.
            Tuple 2 -
                1) A Bx1 label tensor containing 0/1 prediction labels for relevance tasks
                2) A Bx1 weight tensor containing weights for relevance task. They are all set to 1 at the moment.

        """
        # Engagement training data. Assert some expected columns
        is_repin_column = SearchLabels.REPIN.value
        assert is_repin_column in example, f"Invalid engagement data did not include is_repin column: {example}"
        # Extract batch size from known existing column.
        batch_size = len(example[is_repin_column])
        if example.get(LOGGED_RELEVANCE_SCORE_LABEL) is not None:
            relevance_labels = example.get(LOGGED_RELEVANCE_SCORE_LABEL).squeeze()
            relevance_weights = torch.ones(batch_size, device=self.device)
        else:
            relevance_labels = torch.zeros(batch_size, device=self.device)
            relevance_weights = torch.zeros(batch_size, device=self.device)
        engagement_labels = self.is_positive_engagement(example, batch_size)
        engagement_weights = self.get_engagement_weights(example, batch_size, upweight_tprc_factor=upweight_tprc_factor)
        # Above tensors are 1xB so transpose before returning
        return (
            (engagement_labels.T.float(), engagement_weights.T.float()),
            (relevance_labels.T.float(), relevance_weights.T.float()),
        )

    def update_engagement_label_map_lws(self):
        """
        Function to reset the label map for lightweight scoring model.
        :return:
        """
        logger.info("Updating engagement label map for lightweight scoring")
        self.POSITIVE_ENGAGEMENT_LABELS: List[str] = [
            SearchLabels.REPIN.value,
            SearchLabels.LONG_CLICK.value,
            SearchLabels.SAVE_TO_DEVICE.value,
            SearchLabels.CLICK.value,
            SearchLabels.CLOSEUP.value,
        ]
        self.NEGATIVE_ENGAGEMENT_LABELS = [
            SearchLabels.SHORT_CLICK_5S.value,
        ]
        # An ordered list of engagement labels (positive and negative) and their corresponding weights
        # This is ordered so that if a data point meets the first label, it is given the first weight. Otherwise, if it
        # meets the second label, it is given the second weight, and so on.
        self.PRIORITY_ENGAGEMENT_LABEL_TO_WEIGHT: OrderedDict[str, float] = OrderedDict(
            [
                (SearchLabels.LONG_CLICK.value, DEFAULT_TABULAR_LONG_CLICK_WEIGHT_LWS),
                (SearchLabels.SAVE_TO_DEVICE.value, 7.0),
                (SearchLabels.REPIN.value, 7.0),
                (SearchLabels.SHORT_CLICK_5S.value, 20.0),
                (SearchLabels.CLICK.value, 1.0),
                (SearchLabels.CLOSEUP.value, 0.5),
                (SearchLabels.IMPRESSION.value, 5.0),
            ]
        )