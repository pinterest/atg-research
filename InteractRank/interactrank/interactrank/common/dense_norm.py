from __future__ import annotations

from typing import Collection
from typing import Dict

from collections import OrderedDict

import torch
from torch import nn
from interactrank.common.lazy_layernorm import LazyLayerNorm


class BatchDenseNormalization(nn.Module):
    """
    Normalize the list features provided using the normalization layer specified. The module passes through the rest of
    the features.

    Module Initialization:
        :param feats: The list of feature names to normalize.
        :param norm_layer_init_fn: The normalization layer init function to apply.

    Forward:
        :param tensor_dict: The input dictionary of tensors with feature name as key.
        :return: The output dictionary of tensors with normalized dense features.
    """

    def __init__(self, feats: Collection[str], norm_layer_init_fn=LazyLayerNorm):
        super().__init__()

        self.norm_modules = OrderedDict()
        for feat in feats:
            self.norm_modules[_sanitize_name(feat)] = norm_layer_init_fn()
        self.norm_modules = torch.nn.ModuleDict(self.norm_modules)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_tensor_dict = {}
        for feat, tensor in tensor_dict.items():
            module_name = _sanitize_name(feat)
            if module_name in self.norm_modules:
                tensor = self.norm_modules[module_name](tensor.float())
            out_tensor_dict[feat] = tensor
        return out_tensor_dict


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_")
