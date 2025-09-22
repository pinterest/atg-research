from __future__ import annotations

import torch
from torch import nn


class MaskNetBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, project_ratio: float, dropout_ratio: float):
        super(MaskNetBlock, self).__init__()
        self.mask_layer = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim * project_ratio)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim * project_ratio), hidden_dim),
        )

        self.hidden_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim, bias=False),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_ratio),
        )

    def forward(self, V_input, V_hidden):
        V_mask = self.mask_layer(V_input)
        V_output = self.hidden_layer(V_mask * V_hidden)
        return V_output


class ParallelMaskNetLayers(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, project_ratio: float, dropout_ratio: float, block_num: int
    ) -> None:
        super(ParallelMaskNetLayers, self).__init__()
        self.block_num = block_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.project_ratio = project_ratio
        self.dropout_ratio = dropout_ratio
        self.mask_blocks = nn.ModuleList()

    def forward(self, V_emb):
        V_out = V_emb
        block_out = []
        for i in range(self.block_num):
            block_out.append(self.mask_blocks[i](V_emb, V_out))
        concat_out = torch.cat(block_out, dim=-1)
        return concat_out


class LazyParallelMaskNetLayers(nn.modules.lazy.LazyModuleMixin, ParallelMaskNetLayers):
    cls_to_become = ParallelMaskNetLayers

    def __init__(self, output_dim: int, project_ratio: float, dropout_ratio: float, block_num: int):
        super().__init__(
            input_dim=0,
            output_dim=output_dim,
            project_ratio=project_ratio,
            dropout_ratio=dropout_ratio,
            block_num=block_num,
        )
        self.initialize_indicator = nn.UninitializedParameter()

    def initialize_parameters(self, x: torch.Tensor) -> None:
        assert len(x.shape) == 2, x.shape
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.input_dim = x.shape[1]
                self.initialize_indicator.materialize(1)
                for _ in range(self.block_num):
                    self.mask_blocks.append(
                        MaskNetBlock(
                            self.input_dim,
                            hidden_dim=self.input_dim,
                            output_dim=self.output_dim,
                            project_ratio=self.project_ratio,
                            dropout_ratio=self.dropout_ratio,
                        )
                    )
                self.to(self.initialize_indicator.device)
                del self.initialize_indicator
