from __future__ import annotations

from typing import Optional

import torch
from torch.nn import LayerNorm
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter


# Since PyTorch1.9 doesn't have lazy LayerNorm, we implement our own. Deprecate when lazy LayerNorm is implemented in
# future versions of PyTorch
class LazyLayerNorm(LazyModuleMixin, LayerNorm):
    """
    A `LayerNorm` with lazy initialization.

    Module Initialization:
        :param num_normalize_dims:  The number of dimensions to normalize, e.g. 1 for 1d. Default: 1
        :param eps: a value added to the denominator for numerical stability. Default: 1e-5
        :param elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    """

    cls_to_become = LayerNorm

    def __init__(
        self,
        num_normalize_dims: int = 1,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = torch.float,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(0, eps=eps, elementwise_affine=elementwise_affine, **factory_kwargs)
        self.factory_kwargs = factory_kwargs
        self.num_normalize_dims = num_normalize_dims
        if self.elementwise_affine:
            self.weight = UninitializedParameter()
            self.bias = UninitializedParameter()
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.normalized_shape != (0,):
            super().reset_parameters()

    def initialize_parameters(self, x) -> None:
        assert len(x.size()) >= self.num_normalize_dims + 1
        self.normalized_shape = tuple(x.size()[-self.num_normalize_dims :])
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.weight.materialize(self.normalized_shape, **self.factory_kwargs)
                self.bias.materialize(self.normalized_shape, **self.factory_kwargs)
            self.reset_parameters()
