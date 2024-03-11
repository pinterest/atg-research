from __future__ import annotations

from typing import Dict
from typing import Optional

import numpy as np
import torch
from torch import nn


def visual_bytes_to_float(visual: torch.ByteTensor, lookup: torch.FloatTensor) -> torch.Tensor:
    """
    convert a uint8 tensor of shape (*, D) into a f32 tensor of shape (*, 8 * D)
    lu should take care of all the mixed precision casting issues
    """
    # torch doesn't work with unpacking arbitrary input shapes
    if len(visual.shape) == 2:
        return lookup[visual.long()].view(visual.size(0), visual.size(1) * 8)
    elif len(visual.shape) == 3:
        return lookup[visual.long()].view(visual.size(0), visual.size(1), visual.size(2) * 8)
    else:
        raise ValueError("only support visual ndim=3 or 2")


def get_lookup8_numpy():
    """
    Get a (256, 8) read-only ndarray to help convert a uint8 embedding into a f32 embedding of 0 and 1
    """
    out = np.unpackbits(np.arange(0, 256, dtype=np.uint8)).reshape(256, 8)
    out.flags.writeable = False
    return out


class SageEmbedder(nn.Module):
    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class FeatureEmbedder(SageEmbedder):
    def __init__(self, key: str, precision: Optional[torch.dtype] = None):
        super().__init__()
        self.key = key
        self.precision = precision

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        result = feats[self.key]
        if self.precision is not None:
            result = result.to(self.precision)
        return result

    def extra_repr(self) -> str:
        return f"key={self.key}"


class VisualFeatureEmbedder(SageEmbedder):
    def __init__(self, key: str, use_unused_parameter: bool = True):
        super().__init__()

        self.key = key
        self.register_buffer("lookup", torch.tensor(get_lookup8_numpy()).float())
        # so we can create an optimizer
        if use_unused_parameter:
            self.register_parameter("unused", nn.Parameter(torch.zeros(1), requires_grad=True))

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        embs = feats[self.key]
        return visual_bytes_to_float(embs, self.lookup)
