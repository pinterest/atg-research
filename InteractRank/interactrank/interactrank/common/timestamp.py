from __future__ import annotations

from typing import Sequence

import math

import torch
from torch import nn

PADDING_TIMESTAMP = 0


class TimestampEncoder(nn.Module):
    # abs_timestamp_period_hrs: torch.Tensor
    # rel_freq_hrs: torch.Tensor
    # phases: torch.Tensor

    def __init__(
        self,
        abs_timestamp_period_hrs: Sequence[float],
        num_rel_freqs: int,
        max_freq_granularity_h: float,
        min_freq_granularity_h: float,
    ):
        """
        processes timestamps using sin/cos transforms and log on top of 3 time features:
        * absolute time
        * time since previous action
        * time since latest action

        Args:
            abs_timestamp_period_hrs: for each element `period`, we add features
                sin(2 * pi * abs_time / (3600 * 1000 * period))
                cos(2 * pi * abs_time / (3600 * 1000 * period))
            num_rel_freqs:  for num_rel_freqs evenly spaced (in log space) periods between min_freq_granularity_h
                and max_freq_granularity_h, we compute 2 * pi * time_diff / (3600 * 1000 * period) for both
                time difference features, then transform it with both sin and cos to get 2 * num_rel_freqs features
                representing a given sequence element
            max_freq_granularity_h:
            min_freq_granularity_h:
        """
        super().__init__()
        self._padding_timestamp: torch.jit.Final[int] = PADDING_TIMESTAMP

        self.output_dim = 66
        self.register_buffer(
            "abs_timestamp_period_hrs",
            torch.tensor(list(abs_timestamp_period_hrs), dtype=torch.float),
        )
        self.register_buffer(
            "rel_freq_hrs",
            torch.linspace(math.log(min_freq_granularity_h), math.log(max_freq_granularity_h), num_rel_freqs).exp_(),
        )
        self.phases = nn.Parameter(torch.zeros(self.rel_freq_hrs.size(0) + self.abs_timestamp_period_hrs.size(0)))

    def forward(self, timestamps: torch.Tensor, seq_lengths: torch.Tensor = None, request_time: torch.Tensor = None):
        seconds_since_latest_action = (timestamps[:, :1] - timestamps) / 1000.0

        seconds_since_latest_action[timestamps == self._padding_timestamp] = self._padding_timestamp
        seconds_since_latest_action.unsqueeze_(2)
        hours_since_latest_action = seconds_since_latest_action / 3600.0
        all_times = (
            2
            * math.pi
            * torch.cat(
                (
                    (hours_since_latest_action / self.rel_freq_hrs),
                    (timestamps.unsqueeze(2) / (3600_000.0 * self.abs_timestamp_period_hrs)),
                ),
                dim=2,
            )
        )
        # only apply phases to the relative time; unclear exactly why but keeping for snapshot compatibility
        all_times = all_times + self.phases

        sins = torch.sin(all_times)
        coss = torch.cos(all_times)

        return [
            sins,
            coss,
            torch.log(1 + seconds_since_latest_action),
            torch.log(1 + timestamps.unsqueeze(2) / 1000.0),
        ]


class UserSeqTimestampEncoder(nn.Module):
    def __init__(self, base: float = 2.0, bucket_len: int = 12, output_dim: int = 72):
        super().__init__()
        self.register_buffer(
            "boundaries_hr",
            # adding one additional bucket for masking empty timestamp for timestamp embedding
            torch.cat((torch.Tensor([0]), torch.exp(torch.arange(bucket_len))), dim=0),
        )
        self.time_emb = nn.Embedding(num_embeddings=bucket_len + 2, embedding_dim=output_dim)
        self.output_dim = output_dim

    def forward(self, timestamps: torch.Tensor):
        ts_hr = timestamps / 3600_000
        ts_id = torch.bucketize(ts_hr, self.boundaries_hr)
        return self.time_emb(ts_id)
