from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List
from typing import Set

from collections import OrderedDict

import simplejson as json
import torch
from torch import nn
from interactrank.common.tc_compact import TfLazyLinear

NUM_SUMMARIZING_MLP_LAYERS = 2


class MLPSummarizationLayer(nn.Module):
    """
    Summarizes a list of tensors with a MLP network

    Module Initialization:
        :param dim: The shape of the tensors after summarization.
        :param dropout: The dropout probability applied to each group's output tensor.
        :param num_layers: Number of layers in the MLP network
        :param norm_layer_init_fn: Callable returning the norm layer for the module. Default: nn.Identity
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        num_layers: int = NUM_SUMMARIZING_MLP_LAYERS,
        norm_layer_init_fn: Callable[[], nn.Module] = nn.Identity,
    ):
        super(MLPSummarizationLayer, self).__init__()
        layers = OrderedDict()
        for i in range(num_layers):
            layers[f"dense_{i}"] = TfLazyLinear(dim)
            layers[f"activation_{i}"] = torch.nn.LeakyReLU(negative_slope=0.2)
            layers[f"norm_{i}"] = norm_layer_init_fn()
        layers["dropout"] = torch.nn.Dropout(p=dropout)
        self.module = nn.Sequential(layers)

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        :param tensors: List of tensors to summarize each of shape [B, D1]
        :return: Summarized tensor of shape [B, D]
        """
        return self.module(torch.cat(tensors, dim=-1))


class ConcatSummarizationLayer(nn.Module):
    """
    Summarizes a list of tensors with by concatenation followed by an optional norm layer

    Module Initialization:
        :param norm_layer_init_fn: Callable returning the norm layer for the module. Default: nn.Identity
    """

    def __init__(
        self,
        norm_layer_init_fn: Callable[[], nn.Module] = nn.Identity,
    ):
        super(ConcatSummarizationLayer, self).__init__()
        self.norm = norm_layer_init_fn()

    def forward(self, tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        :param tensors: List of tensors to summarize each of shape [B, D1]
        :return: Summarized tensor of shape [B, D]
        """
        return self.norm(torch.cat(tensors, dim=-1))


class BatchSummarization(torch.nn.Module):
    """
    Group features based on the specification and summarize them. For each group, the module fetches each feature's
    tensor from the input tensor dict, concatenate them together into a single tensor and pass them through the group's
    summarization layer. The output tensor is then populated in the output tensor dict with the group name as the key.
    A special case is when a group has only a single feature, the module will simply pass through the input tensor to
    the output tensor dict with group name as its key. The module will also simply pass through features that are not
    listed in the groups to the output tensor dict.

    Module Initialization:
        :param group_to_feats: The dict of group name to its list of feature names that should be summarized.
        :param group_to_layer_init_fn: The dict of group name to its output tensor size after summarization.

    Forward:
        :param tensor_dict: The input dictionary of tensors with feature name as key.
        :return: The output dictionary of tensors.
    """

    def __init__(
        self,
        group_to_feats: Dict[str, List[str]],
        group_to_layer_init_fn: Dict[str, Callable[[], nn.Module]],
    ):
        super().__init__()

        self.group_to_feats = group_to_feats

        self.feats_to_process = _get_all_feats(group_to_feats)

        self.summarizing_modules = OrderedDict()
        for group, summarization_layer_init_fn in group_to_layer_init_fn.items():
            if len(group_to_feats[group]) > 1:
                # Only create layers for feature groups that contain multiple features
                self.summarizing_modules[_sanitize_name(group)] = summarization_layer_init_fn()
        self.summarizing_modules = torch.nn.ModuleDict(self.summarizing_modules)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_tensor_dict = {k: v for k, v in tensor_dict.items() if k not in self.feats_to_process}

        # Group features and perform summarization
        for group, feats in self.group_to_feats.items():
            tensors = [tensor_dict[f] for f in feats]
            if len(tensors) == 1:
                # Passthrough the tensor for feature groups that contain only a single feature
                out_tensor_dict[group] = tensors[0]
            else:
                module = self.summarizing_modules[_sanitize_name(group)]
                out_tensor_dict[group] = module(tensors)

        return out_tensor_dict

    def extra_repr(self) -> str:
        return f"group_to_feats={json.dumps({k: sorted(v) for k, v in self.group_to_feats.items()}, indent=4, sort_keys=True)}"


class BatchMLPSummarization(nn.Module):
    """
    BatchSummarization with MLPSummarization for all groups

    Module Initialization:
        :param group_to_feats: The dict of group name to its list of feature names that should be summarized.
        :param group_to_out_dim: The dict of group name to its output tensor size after summarization.
        :param dropout: The dropout probability applied to each group's output tensor.
        :param norm_layer_init_fn: Callable returning the norm layer for the module. Default: nn.Identity
        :param groups_to_share: Specify groups that share the same summarization layer

    Forward:
        :param tensor_dict: The input dictionary of tensors with feature name as key.
        :return: The output dictionary of tensors.
    """

    def __init__(
        self,
        group_to_feats: Dict[str, List[str]],
        group_to_out_dim: Dict[str, int],
        dropout: float = 0.0,
        norm_layer_init_fn: Callable[[], nn.Module] = nn.Identity,
        num_mlp_layers: int = NUM_SUMMARIZING_MLP_LAYERS,
        groups_to_share: List[List[str]] = None,
    ):
        super().__init__()
        group_to_layer_init_fn = {}
        group_to_shared_group = {}
        shared_group_to_layer = {}

        def init_func(group, **kwargs):
            def _func():
                shared_group = group_to_shared_group.get(group)
                if shared_group is None:
                    return MLPSummarizationLayer(**kwargs)

                mlp = shared_group_to_layer.get(shared_group)
                if mlp is None:
                    mlp = shared_group_to_layer[shared_group] = MLPSummarizationLayer(**kwargs)

                return mlp

            return _func

        if groups_to_share is not None:
            for groups in groups_to_share:
                dim = group_to_out_dim[groups[0]]
                assert all(group_to_out_dim[x] == dim for x in groups)

                hashable_groups = tuple(groups)
                for group in groups:
                    group_to_shared_group[group] = hashable_groups

        for group, dim in group_to_out_dim.items():
            group_to_layer_init_fn[group] = init_func(
                group, dim=dim, dropout=dropout, norm_layer_init_fn=norm_layer_init_fn, num_layers=num_mlp_layers
            )

        self.module = BatchSummarization(
            group_to_feats=group_to_feats,
            group_to_layer_init_fn=group_to_layer_init_fn,
        )

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.module(tensor_dict)


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_")


def _get_all_feats(group_to_feats: Dict[str, List[str]]) -> Set[str]:
    return set([f for feats in group_to_feats.values() for f in feats])
