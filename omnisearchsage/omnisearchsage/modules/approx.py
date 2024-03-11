from __future__ import annotations

from typing import Optional
from typing import Tuple

import logging
import random

import torch
from omnisearchsage.modules.negatives import all_gather_1d_tensor
from torch import nn

LOG = logging.getLogger(__file__)


@torch.jit.script
def hash_func(longs: torch.Tensor, w: int, hash_a: torch.Tensor) -> torch.Tensor:
    # shape: (...) -> (..., self.d)
    # copied from https://github.com/apache/spark/blob/0494dc90af48ce7da0625485a4dc6917a244d580/common/sketch/src/main/java/org/apache/spark/util/sketch/CountMinSketchImpl.java#L203  # noqa
    PRIME_MODULUS = (1 << 31) - 1
    hash_ = longs.unsqueeze(-1) * hash_a
    hash_ += hash_ >> 32
    hash_ &= PRIME_MODULUS
    return (hash_.int() % w).long()


class Counter(nn.Module):
    def update(self, longs: torch.Tensor, increment: int = 1) -> None:
        raise NotImplementedError()

    def forward(self, longs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()


class CountMinSketch(Counter):
    def extra_repr(self) -> str:
        return f"w={self.w},d={self.d}"

    def __init__(self, w: int, d: int, seed: Optional[int] = None, synchronize_counts: bool = False):
        """
        Implements count-min sketch, expecting inputs that are of dtype int64
        Args:
            w: width of sketch
            d: depth of sketch
            seed: random seed to use for hash function initialization. must be specified and the same across all
                processes if synchronize_counts=True
            synchronize_counts: if True, gathers inputs from all processes in `update`
        """
        super().__init__()
        if synchronize_counts and (seed is None):
            raise ValueError("if synchronize_counts is True, seed must be set")
        self.synchronize_counts = synchronize_counts
        self.w = w
        self.d = d
        LOG.info(f"initializing count-min sketch with width={self.w}, depth={self.d}")
        self.register_buffer("counts", torch.zeros((self.d, self.w), dtype=torch.long))
        self.register_buffer("hash_a", torch.zeros(self.d, dtype=torch.long))
        self.register_buffer("idx", torch.arange(self.d))
        self.register_buffer("num_seen", torch.tensor(0))

        r = random.Random(seed)
        for i in range(self.d):
            self.hash_a[i] = r.randint(1, torch.iinfo(torch.long).max)

    def _hash(self, longs: torch.Tensor) -> torch.Tensor:
        return hash_func(longs=longs, w=self.w, hash_a=self.hash_a)

    def update(self, longs: torch.Tensor, increment: int = 1):
        """
        Increments the ids in `longs` by `increment` inside the sketch
        """
        # assert self.training
        hashes = self._hash(longs.view(-1))  # (product(longs.shape), self.d)
        if self.synchronize_counts:
            assert hashes.ndim == 2, hashes.shape
            hashes = torch.cat(all_gather_1d_tensor(hashes.view(-1)), dim=0).view(-1, self.d)
        self.counts.index_put_((self.idx, hashes), torch.full_like(hashes, fill_value=increment), accumulate=True)
        self.num_seen += hashes.size(0) * increment

    @torch.no_grad()
    def forward(self, longs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hashes = self._hash(longs)
        min_cts = self.counts[self.idx, hashes].min(-1).values
        return min_cts, self.num_seen
