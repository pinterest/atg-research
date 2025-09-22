from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

from collections import defaultdict

import numpy as np
import simplejson as json
import torch
from torch import nn

ONE_HUNDRED_YEARS_IN_SEC = 60 * 60 * 24 * 365 * 100


# We have to script parts where tensors are created on the fly due to tracing not able to handle changing of device
# during inference. More information at: https://github.com/pytorch/pytorch/issues/31141.
@torch.jit.script  # type: ignore
def _create_indices(indices: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(indices, dtype=torch.long, device=device)


@torch.jit.script  # type: ignore
def _zeros_like(tensor: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(tensor)


class AutoMLContinuousNormalization(nn.Module):
    """
    Merge and normalize continuous features in the input tensor dict into a single feature. The module drops all
    features that have small standard deviation and handles the normalization by:
        1) feat = clip(feat, min, max) - min             for all feat in which its min_max_range <= 2.0
        2) feat = log(clip(feat, min, max) - min + 1.0)    otherwise
    The merged features are then passed through batch normalization and persisted in the output tensor dict with
    "continuous" as the key.

    Module Initialization:
        :param feature_names: The list of continuous feature names to normalize. The module will use this list to match
            against the keys in the tensor_dict in the forward pass.
        :param min_values_stat: The minimum value of each feature.
        :param max_values_stat: The maximum value of each feature.
        :param stddev_stat: The stddev of each feature.
        :param small_stddev_threshold: The threshold of stddev to drop feature
        :param norm_layer_init_fn: Callable returning the norm layer for the module
        :param fill_nan_values: If True, fill NaN values with 0.0
        :param use_mixture_transformation: if True, we learn a mixture of transformations to be applied per numerical feature
            as proposed in https://research.google/pubs/pub49171/

    Forward:
        :param tensor_dict: The input dictionary of tensors with feature name as key.
        :return: The output dictionary of tensors with continuous features merged into a single feature.
    """

    LOG_TRANSFORM_THRESHOLD = 2.0
    min_values: torch.Tensor
    max_values: torch.Tensor
    min_max_ranges: torch.Tensor
    keeping_mask: torch.Tensor

    def __init__(
        self,
        feature_names: List[str],
        min_values_stat: List[float],
        max_values_stat: List[float],
        stddev_stat: List[float],
        small_stddev_threshold: float = 1e-6,
        explicit_scaling: bool = False,
        norm_layer_init_fn: Callable[[], nn.Module] = nn.LazyBatchNorm1d,
        fill_nan_values: bool = False,
        mean_values_stat: Optional[List] = None,
        use_mixture_transformation: bool = False,
    ):
        super().__init__()

        self.small_stddev_threshold = small_stddev_threshold

        # Initialize sub components used for normalization
        self._initialize_stats_buffers(feature_names, min_values_stat, max_values_stat, stddev_stat, mean_values_stat)
        self.feature_names = [f for i, f in enumerate(feature_names) if self.keeping_mask[i]]
        self.feats_to_drop = [f for i, f in enumerate(feature_names) if not self.keeping_mask[i]]
        self.explicit_scaling = explicit_scaling
        self.norm_layer = norm_layer_init_fn()
        self.fill_nan_values = fill_nan_values

        self.use_mixture_transformation = use_mixture_transformation
        if use_mixture_transformation:
            assert mean_values_stat is not None, "mean_values_stat must be provided when using mixture_transformation"
            self.norm_layer_after_transform = nn.LazyBatchNorm1d()
            self.n_features = len(self.feature_names)
            self.cont_embeddings = nn.Parameter(torch.Tensor(self.n_features, 8))
            nn.init.xavier_normal_(self.cont_embeddings)
            # since we learn a mixture of 3 transformations: raw, log_1p, and gaussian
            self.linear_mixer = nn.Sequential(nn.LazyLinear(3, bias=False), nn.Softmax())

    def _initialize_stats_buffers(
        self,
        feature_names: List[str],
        min_vals: List[float],
        max_vals: List[float],
        std_vals: List[float],
        mean_vals: Optional[List],
    ) -> None:
        min_values = torch.as_tensor(min_vals, dtype=torch.float)
        max_values = torch.as_tensor(max_vals, dtype=torch.float)
        std_values = torch.as_tensor(std_vals, dtype=torch.float)
        feats = np.array(feature_names)

        min_max_range = max_values - min_values
        assert (min_max_range >= 0).all()
        # Ignoring features with small STD for robustness:
        # this is done by always capping the shifted feature value to 0.
        keeping_mask = std_values >= self.small_stddev_threshold

        print(
            f"Ignoring features with small stdev: "
            f"{[(feat_i, f'std={std_i}') for feat_i, std_i in zip(feats[(~keeping_mask).numpy()], std_values[~keeping_mask])]}"
        )

        apply_log_transform = min_max_range > self.LOG_TRANSFORM_THRESHOLD

        print(f"Skipping log_1p transform on inputs : {feats[(keeping_mask & ~apply_log_transform).numpy()]}")
        print(f"Applying log_1p transform on inputs : {feats[(keeping_mask & apply_log_transform).numpy()]}")

        self.register_buffer("min_values", min_values[keeping_mask], False)
        self.register_buffer("max_values", max_values[keeping_mask], False)
        self.register_buffer("std_values", std_values[keeping_mask], False)
        self.register_buffer("min_max_ranges", min_max_range[keeping_mask], False)
        self.register_buffer("keeping_mask", keeping_mask, False)
        if mean_vals is not None:
            mean_values = torch.as_tensor(mean_vals, dtype=torch.float)
            self.register_buffer("mean_values", mean_values[keeping_mask], False)

    def forward(self, tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out_tensor_dict = {
            k: v for k, v in tensor_dict.items() if (k not in self.feature_names and k not in self.feats_to_drop)
        }

        if not (hasattr(self, "indices") and hasattr(self, "dtypes") and hasattr(self, "ft_index_to_dtype_index")):
            self.dtypes = sorted(
                list(set([tensor_dict[feature_name].dtype for feature_name in self.feature_names])),
                key=lambda x: str(x),
            )
            self.ft_index_to_dtype_index = {}  # which tensor group to include the feature in
            i_by_dtype = defaultdict(lambda: [])  # collect feature indices (from self.feature_names) by dtype
            for i, feature_name in enumerate(self.feature_names):
                tensor = tensor_dict[feature_name]
                self.ft_index_to_dtype_index[i] = self.dtypes.index(tensor.dtype)
                i_by_dtype[tensor.dtype].append(i)

            indices = [i for dtype in self.dtypes for i in i_by_dtype[dtype]]
            self.register_buffer("indices", _create_indices(indices, tensor_dict[self.feature_names[0]].device), False)

        grouped_fts = [[] for _ in range(len(self.dtypes))]
        for i, feature_name in enumerate(self.feature_names):
            tensor = tensor_dict[feature_name]
            assert tensor.dim() == 1
            grouped_fts[self.ft_index_to_dtype_index[i]].append(tensor)

        continuous_tensors_unordered = torch.cat([torch.stack(group, dim=0).float() for group in grouped_fts], dim=0).T
        continuous_tensors_raw = _zeros_like(continuous_tensors_unordered)
        continuous_tensors_raw[:, self.indices] = continuous_tensors_unordered

        continuous_tensors = self._normalize_tensors(continuous_tensors_raw)
        if "continuous" in out_tensor_dict:
            # If there is already a "continuous" entry, e.g. from _UniversalNumericPreprocessorModule, add in
            # this module's tensors to that entry as separate columns instead of overwriting it.
            out_tensor_dict["continuous"] = torch.cat([out_tensor_dict["continuous"], continuous_tensors], dim=1)
        else:
            if self.use_mixture_transformation:
                # apply raw, log1p, gaussian transformations
                cont_tensors_raw = torch.clip(continuous_tensors_raw, min=self.min_values, max=self.max_values)
                cont_tensors_log1p = continuous_tensors
                cont_tensors_gaussian = (cont_tensors_raw - self.mean_values) / self.std_values
                # stack the set of transformations
                cont_tensors_all = torch.stack((cont_tensors_raw, cont_tensors_log1p, cont_tensors_gaussian), dim=-1)
                # apply Numerical Feature Transformation (NFT) Mixer to the set of transformations
                out_tensor_dict["continuous"] = torch.einsum(
                    "ijk,jk->ij", cont_tensors_all, self.linear_mixer(self.cont_embeddings)
                )
                # apply BN to the mixture of transformations
                out_tensor_dict["continuous"] = self.norm_layer_after_transform(out_tensor_dict["continuous"])
            else:
                out_tensor_dict["continuous"] = continuous_tensors

        return out_tensor_dict

    def _normalize_tensors(self, x: torch.Tensor) -> torch.Tensor:
        # Fill NaN values with 0
        if self.fill_nan_values:
            x = x.nan_to_num()
        # Clip the tensor by the min/max
        x = torch.clip(x, min=self.min_values, max=self.max_values) - self.min_values
        # Perform log transformation
        x = torch.where(self.min_max_ranges > self.LOG_TRANSFORM_THRESHOLD, torch.log(1.0 + x), x)
        # Scale all values between [0, 1] if explicit_scaling=True
        if self.explicit_scaling:
            x = torch.where(
                self.min_max_ranges > self.LOG_TRANSFORM_THRESHOLD, x / torch.log(1.0 + self.min_max_ranges), x
            )

        return self.norm_layer(x)

    def extra_repr(self) -> str:
        return (
            f"feature_names={json.dumps(sorted(self.feature_names), indent=4)}, "
            f"feats_to_drop={json.dumps(sorted(self.feats_to_drop), indent=4)}"
        )
