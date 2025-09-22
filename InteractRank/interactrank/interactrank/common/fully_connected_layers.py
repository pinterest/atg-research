from __future__ import annotations

from typing import Callable
from typing import List
from typing import Optional
from typing import Sequence
from typing import cast

import math

import torch
from torch import nn
from interactrank.common.lazy_layernorm import LazyLayerNorm
from interactrank.common.tc_compact import TfLazyLinear


class FullyConnectedLayer(torch.nn.Module):
    """
    Generic dense layer.

    Module Initialization:
        :param hidden_size: Hidden size.
        :param dropout: dropout probability for dense layer. Default is 0 (no dropout).
        :param relu_after_layer_norm: If true, apply relu after layer norm.
        :param activation_init_fn: Activation init function to use. Default is ReLU.
        :param low_rank: Rank of the low rank approximation.
        :param norm_layer_init_fn: The normalization layer to use. Default is LazyLayerNorm.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        relu_after_layer_norm: bool = False,
        activation_init_fn: Callable[[], nn.Module] = nn.ReLU,
        low_rank: Optional[int] = None,
        norm_layer_init_fn: nn.Module = LazyLayerNorm,
    ):
        super().__init__()
        if low_rank:
            linear = LazyLowRankLinear(hidden_size, low_rank)
        else:
            linear = TfLazyLinear(hidden_size)

        if relu_after_layer_norm:
            self.layers = nn.Sequential(linear, norm_layer_init_fn(), activation_init_fn())
        else:
            self.layers = nn.Sequential(linear, activation_init_fn(), norm_layer_init_fn())
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class FullyConnectedLayers(torch.nn.Module):
    """
    Generic module to pass the preprocessed tensor through multiple dense layers.

    Module Initialization:
        :param output_size: Size of the output layer
        :param hidden_sizes: The list of dense layer sizes to initialize those layers.
        :param dropout: dropout probability for dense layer. Default is 0 (no dropout).
        :param relu_after_layer_norm: If true, apply relu after layer norm.
        :param activation_init_fn: Activation init function to use. Default is ReLU.
        :param has_skip: If true, apply skip connections between layers.
        :param low_ranks: The list of low ranks for low rank approximation.
        :param norm_layer_init_fn: The normalization layer to use. Default is LazyLayerNorm.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(
        self,
        output_size: int,
        hidden_sizes: Sequence[int],
        dropout: float = 0.0,
        relu_after_layer_norm: bool = False,
        activation_init_fn: Callable[[], nn.Module] = nn.ReLU,
        has_skip: bool = False,
        low_ranks: Optional[Sequence[Optional[int]]] = None,
        norm_layer_init_fn: Optional[Callable[[], nn.Module]] = LazyLayerNorm,
    ):
        super().__init__()
        fc_layer = FullyConnectedWithSkipLayer if has_skip else FullyConnectedLayer
        if low_ranks is None:
            low_ranks = [None] * len(hidden_sizes)
        layers = cast(
            List[nn.Module],
            [
                fc_layer(
                    hidden_size=hidden_size,
                    dropout=dropout,
                    relu_after_layer_norm=relu_after_layer_norm,
                    activation_init_fn=activation_init_fn,
                    low_rank=low_rank,
                    norm_layer_init_fn=norm_layer_init_fn,
                )
                for low_rank, hidden_size in zip(low_ranks, hidden_sizes)
            ],
        ) + cast(List[nn.Module], [TfLazyLinear(output_size)])
        self.layers = nn.Sequential(*layers)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)


class FullyConnectedWithSkipLayer(torch.nn.Module):
    """
    Generic dense layer with skip connections.

    Module Initialization:
        :param hidden_size: Hidden size.
        :param dropout: dropout probability for dense layer. Default is 0 (no dropout).
        :param relu_after_layer_norm: If true, apply relu after layer norm.
        :param activation_init_fn: Activation init function to use. Default is ReLU.
        :param low_rank: Rank of the low rank approximation.
        :param norm_layer_init_fn: The normalization layer to use. Default is LazyLayerNorm.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.0,
        relu_after_layer_norm: bool = False,
        activation_init_fn: Callable[[], nn.Module] = nn.ReLU,
        low_rank: Optional[int] = None,
        norm_layer_init_fn: Optional[Callable[[], nn.Module]] = LazyLayerNorm,
    ):
        super().__init__()
        if low_rank:
            self.linear = LazyLowRankLinear(hidden_size, low_rank)
        else:
            self.linear = TfLazyLinear(hidden_size)
        if relu_after_layer_norm:
            self.layers = nn.Sequential(norm_layer_init_fn(), activation_init_fn())
        else:
            self.layers = nn.Sequential(activation_init_fn(), norm_layer_init_fn())
        if dropout > 0:
            self.layers.append(nn.Dropout(p=dropout))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        assert len(tensor.shape) == 2
        tensor = torch.cat([tensor, self.linear(tensor)], dim=1)
        return self.layers(tensor)


class LowRankLinear(nn.Module):
    """
    Low Rank Linear Layer.

    Module Initialization:
        :param in_features: Size of each input sample.
        :param out_features: Size of each output sample.
        :param rank: Rank of the low rank approximation.
        :param bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.u_weight = nn.Parameter(torch.empty((in_features, rank), **factory_kwargs))
        self.v_weight = nn.Parameter(torch.empty((rank, out_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features), **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.u_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_weight, a=math.sqrt(5))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.addmm(self.bias, mat1=torch.mm(tensor, self.u_weight), mat2=self.v_weight)


class LazyLowRankLinear(nn.modules.lazy.LazyModuleMixin, LowRankLinear):
    """
    Low Rank Linear Layer with lazy initialization.

    Module Initialization:
        :param out_features: Size of each output sample.
        :param rank: Rank of the low rank approximation.
        :param bias: If set to ``False``, the layer will not learn an additive bias.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    cls_to_become = LowRankLinear
    u_weight: nn.UninitializedParameter
    v_weight: nn.UninitializedParameter
    bias: nn.UninitializedParameter

    def __init__(self, out_features: int, rank: int, bias: bool = True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(0, 0, rank, bias, **factory_kwargs)
        self.u_weight = nn.UninitializedParameter(**factory_kwargs)
        self.v_weight = nn.UninitializedParameter(**factory_kwargs)
        self.out_features = out_features
        self.rank = rank
        if bias:
            self.bias = nn.UninitializedParameter(**factory_kwargs)

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.u_weight.materialize((self.in_features, self.rank))
                self.v_weight.materialize((self.rank, self.out_features))
                if self.bias is not None:
                    self.bias.materialize((self.out_features,))
                self.reset_parameters()


class AutoMLFullyConnectedLayer(torch.nn.Module):
    """
    AutoML dense layer.

    Module Initialization:
        :param size: Hidden size.
        :param use_gate_layer: If true, use gate layer.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(
        self,
        hidden_size: int,
        use_gate_layer: bool,
        activation: nn.Module = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.use_gate_layer = use_gate_layer
        self.hidden_layer = TfLazyLinear(hidden_size)
        if self.use_gate_layer:
            self.gate_layer = torch.nn.Sequential(TfLazyLinear(hidden_size), torch.nn.Sigmoid())
        if activation:
            self.activation = activation
        else:
            self.activation = nn.SELU()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.hidden_layer(tensor)
        if self.use_gate_layer:
            tensor = tensor * self.gate_layer(tensor)
        return self.activation(tensor)


class AutoMLFullyConnectedLayers(torch.nn.Module):
    """
    AutoML module to pass the preprocessed tensor through multiple dense layers.

    Module Initialization:
        :param hidden_sizes: The list of dense layer sizes to initialize those layers.
        :param use_gate_layer: If true, apply gate layer.

    Forward:
        :param tensor: The input tensor.
        :return: The output tensor.
    """

    def __init__(
        self,
        hidden_sizes: Sequence[int],
        use_gate_layer: bool,
        use_relu: bool = False,
        activation: nn.Module = None,
    ):
        super().__init__()
        if use_relu:
            activation = nn.ReLU()
        self.layers = nn.Sequential(
            *[AutoMLFullyConnectedLayer(hidden_size, use_gate_layer, activation) for hidden_size in hidden_sizes]
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.layers(tensor)
