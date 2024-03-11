from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple

import dataclasses
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from omnisearchsage.common.eval_utils.containers import Metric
from omnisearchsage.common.eval_utils.containers import permute_seq

if TYPE_CHECKING:
    from omnisearchsage.common.eval_utils.containers import EvalEmbeddings
    from omnisearchsage.common.eval_utils.containers import EvaluationMetadata


def draw_qp_qneg_dist_histogram(eval_task: EvaluationTask, num_negative_rows: int = 500):
    """
    draws a histogram of query-positive distances vs query-negative distances. ideally this should
    show very clear separation of positive and negative examples
    """
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    qp_dists = (eval_task.queries.embeddings * eval_task.positives.embeddings).sum(1)
    neg_samp = np.random.choice(a=eval_task.negatives.embeddings.shape[0], size=num_negative_rows, replace=False)
    qn_dists = (
        (
            torch.as_tensor(eval_task.queries.embeddings).float().to(device=device)
            @ torch.as_tensor(eval_task.negatives.embeddings[neg_samp]).float().to(device=device).t()
        )
        .cpu()
        .reshape(-1)
    )

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bins = np.linspace(-1, 1, 100)
    ax.hist(qp_dists, bins=bins, label="query-positive", density=True)
    ax.hist(qn_dists.numpy(), bins=bins, label="query-negative", density=True)
    ax.legend(loc="best")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("count")
    return fig


@lru_cache()
def r_at_k_metric(k: int) -> Metric:
    return Metric(f"recall_{k}", f"Recall@{k}: {{}}", lambda x, w=None: recall_at_k(x, k))


def recall_at_k(positive_ranks: np.ndarray, k: int) -> float:
    """Calculate average Recall@K"""
    return float(np.mean(positive_ranks <= k))


RECALL_1 = r_at_k_metric(1)
RECALL_10 = r_at_k_metric(10)
RECALL_100 = r_at_k_metric(100)


def normalize(arr: np.ndarray, p: float = 2.0, dim: int = 1) -> np.ndarray:
    norm = np.linalg.norm(arr, axis=dim, ord=p, keepdims=True)
    return arr / norm


def calculate_ranks(
    query_features: np.ndarray,
    positive_features: np.ndarray,
    index_features: np.ndarray,
    chunk_size: int = 8,
    topk_cutoff: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, torch.Tensor]:
    """
    Given query_features, ranks positive_features among index_features.
    Assumes query_features[i] corresponds to positive_features[i]

    Also computes the number of times each index of negatives is returned in the top k (topk_cutoff).
    Only counts are returned from this function, where each count is the number of times some signature is retrieved
    at or below `cutoff`. Note that these results are unique, so if a given signature is returned 10 times, it will
    only have one output row

    Args:
        query_features: (N, D)
        positive_features: (N, D)
        index_features: (I, D)
        chunk_size: how large of a chunk to compute at a time
        topk_cutoff: Cutoff to calculate top retrieved indices
        device: torch device to use for computation
    """
    if device is None:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    query_positive_dists = (
        torch.as_tensor((query_features * positive_features).sum(1)[np.newaxis]).float().to(device=device)
    )  # (1, N)
    query_features_t = torch.as_tensor(query_features.T).float().to(device=device)  # (D, N)
    indf = torch.as_tensor(index_features).float().to(device=device)

    out = torch.zeros(query_positive_dists.shape[1], dtype=torch.long)
    topk_chunks = []

    # chunk computation because full matrix of similarities can't fit in gpu memory
    for i in range(0, query_features_t.shape[1], chunk_size):
        query_t_chunk = query_features_t[:, i : i + chunk_size]
        dists_chunk = query_positive_dists[:, i : i + chunk_size]
        qn = indf @ query_t_chunk
        out[i : i + chunk_size] = (qn >= dists_chunk).sum(0)
        chunk = torch.topk(qn, k=topk_cutoff, dim=0).indices
        topk_chunks.append(chunk.t())
    topk_indices = torch.cat(topk_chunks, dim=0)
    _, cts = torch.unique(topk_indices, return_counts=True)

    return out.cpu().numpy() + 1, cts.cpu()


@dataclass
class EvaluationTask:
    """
    Class to pass around all the information associated with extracted embeddings
    """

    # fields that should be saved once per index
    INDEX_FIELDS: ClassVar[Set[str]] = {
        "negatives",
    }

    # fields that should be saved once per set of pairs
    PAIR_FIELDS: ClassVar[Set[str]] = {
        "queries",
        "positives",
        "action_counts",
        "eval_tags",
    }

    queries: EvalEmbeddings = None
    positives: EvalEmbeddings = None
    negatives: EvalEmbeddings = None

    # cache for self.positive_ranks. The @cached_property decorator might be usable instead,
    # but it's only included in py38+. @lru_cache isn't per-instance, so @lru_cache(1) only
    # works if there's only one instance of this class around
    _positive_ranks: np.ndarray = None

    # cache for self.topk_indices_freq
    _topk_indices_freq: torch.Tensor = None
    topk_metrics_cutoff: int = 10

    # if set, evaluation will be based on the first embedding_dim elements of the query, pos, and neg embeddings
    embedding_dim: Optional[int] = None

    def set_embedding_dim(self, dim: int):
        self.embedding_dim = dim
        self._positive_ranks = None
        self._topk_indices_freq = None

    def _calculate_ranks(self):
        if self._positive_ranks is None:
            embeddings = dict(
                query_features=self.queries.embeddings,
                positive_features=self.positives.embeddings,
                index_features=self.negatives.embeddings,
            )
            if self.embedding_dim:
                for k, v in embeddings.items():
                    # might need to do normalization in batches depending on the eval data size
                    embeddings[k] = normalize(v[..., : self.embedding_dim], p=2, dim=-1)
            self._positive_ranks, self._topk_indices_freq = calculate_ranks(
                **embeddings,
                topk_cutoff=self.topk_metrics_cutoff,
            )

    @property
    def positive_ranks(self):
        """
        Compute the ranks of positive examples. Only computed once
        """
        self._calculate_ranks()
        return self._positive_ranks

    @property
    def topk_indices_freq(self):
        self._calculate_ranks()
        return self._topk_indices_freq

    def permute_qp(self, perm):
        """
        Given some subset of a permutation of range(self.queries.embeddings.keys),
        replaces all query/positive attributes of this class with those indexed by this
        permutation
        """

        return dataclasses.replace(
            self,
            queries=self.queries.permute(perm),
            positives=self.positives.permute(perm),
            _positive_ranks=permute_seq(self._positive_ranks, perm),
        )

    def __eq__(self, other) -> bool:
        """
        This is to allow this to be a key of dict
        """
        return self is other

    def __hash__(self) -> int:
        """
        This is to allow this to be a key of a dict
        """
        return hash(id(self))

    # dataclasses don't have such a function; added to keep api consistent with NamedTuple
    def _replace(self, **attrs) -> EvaluationTask:
        return self.replace(**attrs)

    def replace(self, **attrs) -> EvaluationTask:
        return dataclasses.replace(self, **attrs)


@torch.no_grad()
def compute_freq_metrics(meta: Optional[EvaluationMetadata], frequencies: torch.Tensor, cutoff: int):
    r"""
    Computes metrics for some LongTensor of frequencies (returned by compute_retrieved_idx_frequencies). cutoff
    is used to format the metric name e.g. p90_freq_10 where 10 means the frequency was calculated from the top 10
    retrieved results.

    The 3 metrics currently computed are:
    * p99_freq: the 99th percentile of frequencies of retrieved objects. Because there is one entry in frequencies
        per distinct object, we need a repeat_interleave to correctly compute the percentile
    * expected_freq: the expected value of the frequency with which an object is retrieved in the eval set.
        We retrieve frequencies.sum() objects in total, so
        E = sum_{r \in all non-unique retrieved objects} freq(r) / (total number of retrieved objects)
          = sum_{r \in all non-unique retrieved objects} freq(r) / frequencies.sum()
          = sum_{r \in all unique retrieved objects} freq(r) * freq(r) / frequencies.sum()
    * unique_ratio: the ratio of the number of unique objects retrieved to the total number of objects retrieved
    """
    cts = frequencies.float()
    metrics = {
        f"p100_freq_{cutoff}": frequencies.max().item(),
        f"p90_freq_{cutoff}": torch.quantile(cts.repeat_interleave(frequencies, dim=0), 0.90).item(),
    }
    if meta is not None and meta.summary_writer is not None:
        for k, v in metrics.items():
            meta.summary_writer.add_scalar(os.path.join("debug_metrics", meta.prefix, k), v, global_step=meta.iteration)
    return metrics


def write_scalar_metrics(
    metadata: EvaluationMetadata,
    eval_task: EvaluationTask,
    metrics: Sequence[Metric],
) -> Dict[str, Dict[str, Any]]:
    """
    runs `process_one_metric` for a sequence of metrics and returns the result in a dictionary
    """
    out = {}
    for m in metrics:
        out[m.metric_name] = m.compute(eval_task.positive_ranks)
        if metadata is not None and metadata.summary_writer is not None:
            metadata.summary_writer.add_scalar(
                os.path.join(metadata.prefix, m.metric_name),
                out[m.metric_name],
                global_step=metadata.iteration,
            )

    cutoff = 10
    freq_metrics = compute_freq_metrics(
        meta=metadata,
        frequencies=eval_task.topk_indices_freq,
        cutoff=cutoff,
    )
    out["frequency"] = freq_metrics

    return out
