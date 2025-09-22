from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Collection
from typing import Dict
from typing import List
from typing import Optional

import simplejson as json
import torch
from torch import fx
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from typing import NamedTuple

if TYPE_CHECKING:
    from interactrank.common.utils.utils import ExampleType


class RaggedTensorTriplet(NamedTuple):
    """
    The triplet of tensors representing a ragged tensor which is the tensor equivalent of nested variable-length lists.
    We flatten the tensor into a 1-dimensional tensor and store the offsets to each row.

    :param offsets: The tensor containing the offsets of each row. The ids and weights of row i are ids[j] and
        weights[j] for offsets[i] <= j < offsets[i + 1]. Note that its shape is num_rows + 1 where the last element is
        the total number of ids/weights.
    :param ids: The 1-dimensional tensor containing the flatten id tensors.
    :param weights: The 1-dimensional tensor containing the flatten weight tensors.
    """

    offsets: torch.Tensor
    ids: torch.Tensor
    weights: Optional[torch.Tensor] = None

class TracedFormatFeatureRegistryInput(nn.Module):
    """
    Traced version of FormatFeatureRegistryInput based on the batch that is provided. This provides a self-contained
    module that can then be exported to torch::deploy and potentially improve training performance. Users can properly
    initialize the module only through LazyTracedFormatFeatureRegistryInput and the init function of this module is
    intentionally left blank.

    Forward:
        :param tensor_dict: The input dictionary of tensors where the key is the feature name.
        :return: The output dictionary of formatted tensors.
    """

    def __init__(self):
        super().__init__()
        self.traced_module: Optional[fx.GraphModule] = None
        self.expected_keys: Optional[List[str]] = None

    def forward(self, tensor_dict: ExampleType) -> ExampleType:
        assert self.traced_module, (
            "TracedFormatFeatureRegistryInput must be properly initialized through LazyTracedFormatFeatureRegistryInput"
        )
        return self.traced_module(tensor_dict)

    def extra_repr(self) -> str:
        return f"features={json.dumps(self.expected_keys, indent=4)}"


class ConcatInput(nn.Module):
    """Combine dict of tensors into a single tensor feature

    Module Initialization:
        :param expected_keys: The expected keys of the input dictionary of tensors.
        :param cast_dtype: The dtype to cast the tensors to. Setting this can reduce the copy overhead in stacking.
        :param to_skip: The keys to skip from the input dictionary of tensors. Supercedes expected_keys.
    """

    def __init__(
        self,
        expected_keys: Collection[str],
        cast_dtype: Optional[torch.dtype] = None,
        to_skip: Optional[Collection[str]] = None,
    ):
        super(ConcatInput, self).__init__()
        self.to_skip = set(to_skip) if to_skip is not None else set()
        self.expected_keys = list(sorted(set(expected_keys) - self.to_skip))
        self.cast_dtype = cast_dtype

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        assert len(tensor_dict.keys() - self.to_skip) == len(self.expected_keys), (
            tensor_dict.keys() - self.to_skip,
            self.expected_keys,
        )
        return torch.column_stack([tensor_dict[key].to(self.cast_dtype) for key in self.expected_keys])

    def extra_repr(self) -> str:
        return f"features={json.dumps(self.expected_keys, indent=4)}"


class LazyConcatInput(LazyModuleMixin, ConcatInput):
    """Lazy Initialized module for concat input to set expected_keys automatically upon initialization"""

    cls_to_become = ConcatInput

    def __init__(self, cast_dtype: Optional[torch.dtype] = None, to_skip: Optional[Collection[str]] = None):
        super(LazyConcatInput, self).__init__(expected_keys=[], cast_dtype=cast_dtype, to_skip=to_skip)
        self.uninitialized_placeholder = nn.UninitializedParameter()

    def initialize_parameters(self, tensor_dict: Dict[str, torch.Tensor]):
        if self.has_uninitialized_params():
            ragged_tensors = {k: v for k, v in tensor_dict.items() if isinstance(v, RaggedTensorTriplet)}
            assert not len(ragged_tensors), f"ConcatInput does not accept ragged tensors: {set(ragged_tensors.keys())}"
            self.expected_keys = list(sorted(tensor_dict.keys() - self.to_skip))
            del self.uninitialized_placeholder


class FeatureDropOutInput(torch.nn.Module):
    """
    Apply dropout across the specified feature columns.
    Note: the dropout feature columns should have the same dimensionality.
    Module Initialization:
        :param p: The probability of dropping out the specified feature columns
        :param dropout_feature_cols: The list of feature columns to apply dropout to
    """

    def __init__(
        self,
        p: float,
        dropout_feature_cols: List[str],
    ):
        super().__init__()

        self.p = p
        self.dropout_feature_cols = dropout_feature_cols

        if self.p > 0.0:
            assert self.p < 1.0, "dropout probability should be smaller than 1.0, but has {self.p}"

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.training and self.p > 0 and len(self.dropout_feature_cols) > 0:
            dropout_tensor_dict = {k: v for k, v in tensor_dict.items() if k in self.dropout_feature_cols}
            batch_size = next(iter(dropout_tensor_dict.values())).size(0)
            inv_scaling_factor = 1.0 / (1.0 - self.p)

            for k in self.dropout_feature_cols:
                mask = (torch.rand(batch_size) > self.p).float()
                mask *= inv_scaling_factor
                v = dropout_tensor_dict[k]
                dropout_tensor_dict[k] = mask.to(v.device) * v

            tensor_dict.update(dropout_tensor_dict)
        return tensor_dict
