from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union
from typing_extensions import Protocol

import logging
import math
from enum import Enum

import torch
from omnisearchsage.common.logging.visualization import calc_global_iteration
from omnisearchsage.modules.approx import CountMinSketch
from omnisearchsage.modules.negatives import AllGatherWithGrad
from omnisearchsage.modules.negatives import all_gather_1d_tensor
from torch import nn
from torch.nn import functional as F

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.tensorboard import SummaryWriter


LOG = logging.getLogger(__name__)


class LogProbEstimate(Protocol):
    def estimated_log_prob(self) -> Tensor:
        raise NotImplementedError


class NegCounterResults(NamedTuple):
    approx_cts: torch.Tensor  # (N, )
    total_ct: torch.Tensor  # ()
    all_ids: torch.Tensor  # (N, )

    def estimated_log_prob(self) -> Tensor:
        """
        We want to estimate P(n|q), probability that a (q, n) pair is observed, condition on observing q
        Since the batch is iid, P(n|q) = P(n)

        P(neg in batch) = P(neg appears at least once in batch) = 1 - P(neg never appears in batch)
        = 1 - (1 - neg_freq / neg_total)^neg_batch_size.
                  We approximate (1-x)^n ≈ 1 - nx
        \approx 1 - (1 - neg_batch_size * neg_freq / neg_total)
        = neg_batch_size * neg_freq / neg_total
        Note that this n * x > (1 - (1 - x)^n), so this approach leads to overestimating P(neg in batch), giving less
        of a boost to very popular pins
        In practice, skipping this approximation produces similar results on engagement evals but makes
        frequency eval worse

        """
        log_neg_batch_size = math.log(self.approx_cts.size(0)) if self.approx_cts.size(0) > 0 else 0.0
        return torch.log(self.approx_cts) - torch.log(self.total_ct) + log_neg_batch_size

    def apply_mask(self, mask: Tensor) -> NegCounterResults:
        return self._replace(approx_cts=self.approx_cts[mask], all_ids=self.all_ids[mask])


class NegStrFreqResults(NamedTuple):
    str_freq: torch.Tensor
    all_ids: torch.Tensor
    initial_freq: int

    def estimated_log_prob(self) -> Tensor:
        return -torch.log(self.str_freq) + math.log(self.initial_freq, 10)

    def apply_mask(self, mask: Tensor) -> NegStrFreqResults:
        return self._replace(str_freq=self.str_freq[mask], all_ids=self.all_ids[mask])


class QPCounterResults(NamedTuple):
    # let
    #   Q = number of pairs in loss
    #   A = sum(Q) over all ranks if sync, else Q
    qp_freqs: torch.Tensor  # (Q, )
    q_freqs: torch.Tensor  # (Q, )
    q_ids: torch.Tensor  # (Q, )
    p_ids: torch.Tensor  # (Q, )
    all_q_ids: torch.Tensor  # (A, )
    all_p_ids: torch.Tensor  # (A, )

    def estimated_log_prob(self) -> Tensor:
        """
        P(p|q) = P(p, q)/P(q) = P(p, q)/sum_i P(p_i, q) = C(p, q)/sum_i C(p_i, q) = C(p, q)/C(q)
        """
        return torch.log(self.qp_freqs) - torch.log(self.q_freqs)


class EstimatedFreqCounts(NamedTuple):
    conditional_cts: QPCounterResults
    in_batch_neg_cts: Optional[Union[NegCounterResults, NegStrFreqResults]]
    rnd_neg_cts: Optional[Union[NegCounterResults, NegStrFreqResults]]


class NegativeCounter(nn.Module):
    def __init__(self, d: int = 2, w: int = 2**26):
        super().__init__()
        # synchronize_counts=False because we cather ids before updating the CMS
        self.freq_est = CountMinSketch(d=d, w=w, seed=19980115, synchronize_counts=False)

    def forward(self, ids: torch.Tensor, sync: bool) -> NegCounterResults:
        ids = ids.view(-1)
        if sync:
            ids = torch.cat(all_gather_1d_tensor(ids))
        self.freq_est.update(ids)
        approx_cts, total_ct = self.freq_est(ids)
        return NegCounterResults(approx_cts=approx_cts, total_ct=total_ct, all_ids=ids)


class ConditionalProbabilityEstimator(nn.Module):
    def __init__(
        self,
        d: int = 2,
        w: int = 2**26,
    ):
        super().__init__()
        # synchronize_counts=False because we gather ids before updating the CMS
        self.q_freq_est = CountMinSketch(d=d, w=w, seed=19980115, synchronize_counts=False)
        self.qp_freq_est = CountMinSketch(d=d, w=w, seed=19980115, synchronize_counts=False)

    def forward(self, query_ids: torch.Tensor, pos_ids: torch.Tensor, sync: bool) -> QPCounterResults:
        query_ids = query_ids.view(-1)
        pos_ids = pos_ids.view(-1)

        if sync:
            all_query_ids = torch.cat(all_gather_1d_tensor(query_ids))
            all_pos_ids = torch.cat(all_gather_1d_tensor(pos_ids))
        else:
            all_query_ids = query_ids
            all_pos_ids = pos_ids
        # positive estimates (conditional per query)
        self.qp_freq_est.update(all_pos_ids + 17 * all_query_ids)
        self.q_freq_est.update(all_query_ids)

        qp_freqs, _ = self.qp_freq_est(pos_ids + 17 * query_ids)
        q_freqs, _ = self.q_freq_est(query_ids)

        return QPCounterResults(
            qp_freqs=qp_freqs,
            q_freqs=q_freqs,
            q_ids=query_ids,
            p_ids=pos_ids,
            all_q_ids=all_query_ids,
            all_p_ids=all_pos_ids,
        )


class SoftmaxNegativeMode(str, Enum):
    # Use only in-batch negatives and no random negatives
    IN_BATCH = "IN_BATCH"
    # Use only the random negatives
    RANDOM = "RANDOM"
    # Use both random and in-batch negatives, calculate loss of each separately
    MIXED_SUM = "MIXED_SUM"
    # Concatenate both random and in-batch negatives and calculate one loss term
    MIXED_CONCAT = "MIXED_CONCAT"


class LogitsProcessor:
    def __call__(self, logits: torch.Tensor, *, qp_info: LogProbEstimate, neg_info: LogProbEstimate) -> torch.Tensor:
        """

        :param logits:tensor of shape B, (1+N), where N is the number of negatives and the 0th index represents the
            positive pair logit
        :param qp_info: Estimated log probability for P(p|q)
        :param neg_info: Estimated log probability for P(n|q) = P(n) since the batch is sampled iid
        """
        raise NotImplementedError


class PositiveSPCLogitsProcessor(LogitsProcessor):
    def __call__(self, logits: torch.Tensor, *, qp_info: LogProbEstimate, **kwargs) -> torch.Tensor:
        """
        The sample correction methods discounts the positive pair probability with P(p|q).

        :return: Logits with sample correction applied to the positives
        """
        logits[:, 0] -= qp_info.estimated_log_prob()
        return logits


class NegativeSPCLogitsProcessor(LogitsProcessor):
    def __call__(self, logits: torch.Tensor, *, neg_info: LogProbEstimate, **kwargs) -> torch.Tensor:
        """
        :return: Logits with sample probability correction applied to negatives
        """
        logits[:, 1:] -= neg_info.estimated_log_prob()
        return logits


class NegativeAverageLogitsProcessor(LogitsProcessor):
    def __call__(self, logits: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        neg_logits = neg_logits - log(num_neg)
        e^(neg_logits) = e^(neg_logits)/num_neg
        sum(e^(neg_logits))=(1/num_neg) * sum(e^(neg_logits))

        :return: Logits with abg weight applied to negative logits
        """

        log_num_neg = math.log(logits.size(1) - 1) if logits.size(1) > 1 else 0.0
        logits[:, 1:] -= log_num_neg
        return logits


def get_default_logits_processor() -> List[LogitsProcessor]:
    return [NegativeSPCLogitsProcessor(), PositiveSPCLogitsProcessor()]


class SoftmaxCorrectionLoss(nn.Module):
    """
    Softmax with sample probability correction and learned temperature. computes 2 losses:
    1. loss over in-batch positives
    1. loss over random negatives

    Softmax with correction:
        https://www.tensorflow.org/extras/candidate_sampling.pdf

    Temperature scaling learned in log space:
        https://arxiv.org/abs/2103.00020
    """

    def __init__(
        self,
        synchronize_pos: bool = True,
        *,
        w: int,
        d: int,
        summary_prefix: str = "",
        max_in_batch_negatives: Optional[int] = None,
        neg_mode: SoftmaxNegativeMode = SoftmaxNegativeMode.MIXED_SUM,
        enable_spc: bool = True,
        use_stream_freq_est: bool = False,
        share_masks_for_same_query: bool = False,
        synchronize_negs: bool = True,
        inverse_pos_freq_weight: float = 0.0,
        logits_processors: List[LogitsProcessor] = None,
        initial_temperature: float = 0.07,
        return_dict: bool = False,
        **extra_estimator_args: Any,
    ):
        """
        Args:
            synchronize_pos: if true, gathers positives from all processes to use as negatives
            synchronize_negs: if true, gather random negatives from all processes
            w: Count-min sketch width
            d: Count-min sketch depth
            summary_prefix: prefix for writing metrics
            max_in_batch_negatives: maximum number of in batch positives to use as negatives from this rank.
                If synchronize_pos is true, then the total number of negatives will be at most max_in_batch_negatives * world_size,
                and otherwise it'll be max_in_batch_negatives
            enable_spc: if True, uses sample probability correction
            neg_mode: method of sampling negatives. can be in-batch, random, in-batch + random,
                or concat(in-batch, random).
            inverse_pos_freq_weight: apply inverse pos frequency weight if set to a positive number λ, weights in loss
                calculation will be modified as weights *= exp(λ * (1 - pos_freq / pos_freq.median()))
            initial_temperature: initialization value for the learned softmax temperature
            return_dict: if True, returns a dictionary of loss tensors instead of a list of tensors
        """
        super().__init__()
        self.enable_spc = enable_spc
        self.neg_mode = neg_mode
        self.max_in_batch_negatives = max_in_batch_negatives
        self.share_masks_for_same_query = share_masks_for_same_query
        self.inverse_pos_freq_weight = inverse_pos_freq_weight
        self._summary_prefix = summary_prefix.rstrip("/")
        self._synchronize_pos = synchronize_pos
        self._synchronize_negs = synchronize_negs
        self.temperature = nn.Parameter(torch.tensor(-math.log(initial_temperature), requires_grad=True))
        self._init_counters(w=w, d=d, **extra_estimator_args)

        self.register_forward_pre_hook(type(self).clamp_temp)
        self._num_in_batch_negs: int = -1
        self._num_pairs: int = -1
        self._estimated_freq_counts: EstimatedFreqCounts = None
        if logits_processors is None:
            logits_processors = get_default_logits_processor() if enable_spc else []

        self.logits_processor = logits_processors
        self.return_dict = return_dict

    def _init_counters(self, **estimator_args: Any) -> None:
        self.conditional_ct = ConditionalProbabilityEstimator(**estimator_args)
        if self.neg_mode in (SoftmaxNegativeMode.IN_BATCH, SoftmaxNegativeMode.MIXED_SUM):
            self.in_batch_neg_ct = NegativeCounter(**estimator_args)
        if self.neg_mode in (SoftmaxNegativeMode.RANDOM, SoftmaxNegativeMode.MIXED_SUM):
            self.rnd_neg_ct = NegativeCounter(**estimator_args)
        if self.neg_mode == SoftmaxNegativeMode.MIXED_CONCAT:
            self.in_batch_neg_ct = NegativeCounter(**estimator_args)
            self.rnd_neg_ct = NegativeCounter(**estimator_args)

    def extra_repr(self) -> str:
        return f"neg_mode={self.neg_mode.name}, enable_spc={self.enable_spc}, max_in_batch_negatives={self.max_in_batch_negatives}"

    def clamp_temp(self, _) -> None:
        self.temperature.data.clamp_(max=math.log(100.0))

    @staticmethod
    def _share_masks(mask: torch.Tensor, qid: torch.Tensor):
        unique_qid, indices = torch.unique(qid, sorted=False, return_inverse=True)
        unique_mask = torch.zeros(
            (unique_qid.size(0), mask.size(1)), dtype=torch.bool, device=mask.device
        ).scatter_add_(0, indices.unsqueeze(1).expand(-1, mask.size(1)), mask)
        return unique_mask[indices]

    def _mask_logits(
        self,
        logits: torch.Tensor,
        qp_info: QPCounterResults,
        neg_info: Union[NegCounterResults, NegStrFreqResults],
        is_in_batch_neg: bool,
    ) -> torch.Tensor:
        """
        Masks logits so that we don't use true postives as negatives
        """
        # mask negatives that are equal to the positive example
        negatives_to_mask = qp_info.p_ids.unsqueeze(1) == neg_info.all_ids.unsqueeze(0)
        if self.share_masks_for_same_query:
            # all rows with the same q_id should share the same mask
            negatives_to_mask = self._share_masks(negatives_to_mask, qp_info.q_ids)
        if not self.share_masks_for_same_query and is_in_batch_neg:
            # if we are using positives as negatives, and have a query q appear twice (q, p1), (q, p2) in the batch,
            # then p2 should not be a negative for q, even in the row where p1 is positive
            # share_masks already covers the cases for in-batch negatives
            negatives_to_mask |= qp_info.q_ids.unsqueeze(1) == qp_info.all_q_ids.unsqueeze(0)
        logits[:, 1:].masked_fill_(negatives_to_mask, torch.finfo(logits.dtype).min)
        return logits

    def _scale_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * torch.exp(self.temperature)

    def scale_correct_and_mask(
        self,
        logits: torch.Tensor,
        qp_info: QPCounterResults,
        neg_info: Union[NegCounterResults, NegStrFreqResults],
        is_in_batch_neg: bool,
    ) -> torch.Tensor:
        """
        Takes in some unscaled logits, where the first column are true positives, and the remaining columns are the
        potential negative examples (or the positive example repeated), and correctly scales and masks them.
        Step 1: scale by temperature
        Step 2: apply sample probability correction
        Step 3: mask out examples so that true positives are not treated as negatives

        logits might look something like this:
        qa qa qb qc qd
        qb qa qb qc qd
        rb ra rb rc rd
        sf sa sb sc sd
        where a, b, c, d, f are candidates, q, r, s are queries, and qb is q @ b, and a, b, b, f are positives

        Masking (is_in_batch_neg=True) will make this look like
        qa xx xx qc qd
        qb xx xx qc qd
        rb ra xx rc rd
        sf sa sb sc sd
        so true positives aren't treated as negatives
        """
        logits = self._scale_logits(logits)
        for processor in self.logits_processor:
            logits = processor(logits, qp_info=qp_info, neg_info=neg_info)
        logits = self._mask_logits(logits, qp_info=qp_info, neg_info=neg_info, is_in_batch_neg=is_in_batch_neg)
        return logits

    def compute_loss(self, target: torch.Tensor, logits: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        loss = F.cross_entropy(input=logits, target=target, reduction="none" if weights is not None else "mean")

        if weights is not None:
            loss = F.normalize(weights.view(-1), p=1.0, dim=0) @ loss
        return loss

    def compute_similarity_loss(
        self,
        query: torch.Tensor,
        pos: torch.Tensor,
        all_reduce_pos: torch.Tensor,
        all_reduce_rnd_neg: Optional[torch.Tensor],
        weights: torch.Tensor,
        estimated_counts: EstimatedFreqCounts,
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """Compute similarity between query and pos/neg candidates, and then compute probability corrected softmax loss"""
        losses = {}
        qp_sims = (query * pos).sum(1, keepdim=True)
        q_posneg_sims = query @ all_reduce_pos.t()
        if self.neg_mode != SoftmaxNegativeMode.IN_BATCH:
            assert all_reduce_rnd_neg is not None
            q_rndneg_sims = query @ all_reduce_rnd_neg.t()
        # target is the index of logits (always first column due to combination of cat((qp_sims, q_posneg_sims)) and
        # masking
        target = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
        if self.neg_mode in (SoftmaxNegativeMode.IN_BATCH, SoftmaxNegativeMode.MIXED_SUM):
            posneg_logits = self.scale_correct_and_mask(
                torch.cat((qp_sims, q_posneg_sims), dim=1),
                qp_info=estimated_counts.conditional_cts,
                neg_info=estimated_counts.in_batch_neg_cts,
                is_in_batch_neg=True,
            )
            losses["in_batch"] = self.compute_loss(target, logits=posneg_logits, weights=weights)
        if self.neg_mode in (SoftmaxNegativeMode.RANDOM, SoftmaxNegativeMode.MIXED_SUM):
            rndneg_logits = self.scale_correct_and_mask(
                torch.cat((qp_sims, q_rndneg_sims), dim=1),
                qp_info=estimated_counts.conditional_cts,
                neg_info=estimated_counts.rnd_neg_cts,
                is_in_batch_neg=False,
            )
            losses["rnd_neg"] = self.compute_loss(target, logits=rndneg_logits, weights=weights)
        if self.neg_mode == SoftmaxNegativeMode.MIXED_CONCAT:
            neg_counts_merged = NegCounterResults(
                approx_cts=torch.cat(
                    (estimated_counts.in_batch_neg_cts.approx_cts, estimated_counts.rnd_neg_cts.approx_cts), dim=0
                ),
                total_ct=estimated_counts.in_batch_neg_cts.total_ct + estimated_counts.rnd_neg_cts.total_ct,
                all_ids=torch.cat(
                    (estimated_counts.in_batch_neg_cts.all_ids, estimated_counts.rnd_neg_cts.all_ids), dim=0
                ),
            )
            all_logits = self.scale_correct_and_mask(
                torch.cat((qp_sims, q_posneg_sims, q_rndneg_sims), dim=1),
                qp_info=estimated_counts.conditional_cts._replace(
                    all_q_ids=torch.cat(
                        (
                            estimated_counts.conditional_cts.all_q_ids,
                            torch.full_like(estimated_counts.rnd_neg_cts.all_ids, fill_value=-1),
                        ),
                        dim=0,
                    ),
                ),
                neg_info=neg_counts_merged,
                is_in_batch_neg=True,
            )
            losses["mixed_concat"] = self.compute_loss(target, logits=all_logits, weights=weights)
        return losses if self.return_dict else list(losses.values())

    def _sample_in_batch_negatives(self, pos: torch.Tensor) -> Tuple[Tensor, Tensor]:
        if self.max_in_batch_negatives is not None:
            sample_idx = torch.zeros(pos.size(0), device=pos.device, dtype=torch.bool)
            if pos.size(0) > self.max_in_batch_negatives:
                sample_idx[
                    torch.ones(pos.size(0), device=pos.device).multinomial(
                        num_samples=self.max_in_batch_negatives, replacement=False
                    )
                ] = True
            else:
                sample_idx[:] = True
            if self._synchronize_pos:
                all_reduce_pos = torch.cat(AllGatherWithGrad.apply(pos[sample_idx].contiguous()), dim=0)
                mask = torch.cat(all_gather_1d_tensor(sample_idx), dim=0)
            else:
                all_reduce_pos = pos[sample_idx]
                mask = sample_idx
        else:
            if self._synchronize_pos:
                all_reduce_pos = torch.cat(AllGatherWithGrad.apply(pos.contiguous()), dim=0)
            else:
                all_reduce_pos = pos
            mask = None
        return all_reduce_pos, mask

    def _estimate_freq_counts(self, query_ids: Tensor, pos_ids: Tensor, neg_ids: Tensor) -> EstimatedFreqCounts:
        conditional_cts: QPCounterResults = self.conditional_ct(query_ids, pos_ids, sync=self._synchronize_pos)
        in_batch_neg_cts = None
        rnd_neg_cts = None
        if self.neg_mode in (SoftmaxNegativeMode.IN_BATCH, SoftmaxNegativeMode.MIXED_SUM):
            in_batch_neg_cts = self.in_batch_neg_ct(pos_ids, sync=self._synchronize_pos)
        if self.neg_mode in (SoftmaxNegativeMode.RANDOM, SoftmaxNegativeMode.MIXED_SUM):
            rnd_neg_cts = self.rnd_neg_ct(neg_ids, sync=self._synchronize_negs)
        if self.neg_mode == SoftmaxNegativeMode.MIXED_CONCAT:
            in_batch_neg_cts = self.in_batch_neg_ct(pos_ids, sync=self._synchronize_pos)
            rnd_neg_cts = self.rnd_neg_ct(neg_ids, sync=self._synchronize_negs)
        return EstimatedFreqCounts(
            conditional_cts=conditional_cts,
            in_batch_neg_cts=in_batch_neg_cts,
            rnd_neg_cts=rnd_neg_cts,
        )

    def _mask_estimated_counts(
        self, mask: Optional[Tensor], estimated_counts: EstimatedFreqCounts
    ) -> EstimatedFreqCounts:
        if mask is None:
            return estimated_counts
        conditional_cts = estimated_counts.conditional_cts
        conditional_cts = conditional_cts._replace(
            all_p_ids=conditional_cts.all_p_ids[mask],
            all_q_ids=conditional_cts.all_q_ids[mask],
        )
        in_batch_neg_cts = estimated_counts.in_batch_neg_cts
        if in_batch_neg_cts is not None:
            in_batch_neg_cts = in_batch_neg_cts.apply_mask(mask)
        return EstimatedFreqCounts(
            conditional_cts=conditional_cts,
            in_batch_neg_cts=in_batch_neg_cts,
            rnd_neg_cts=estimated_counts.rnd_neg_cts,
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        pos: torch.Tensor,
        query_ids: torch.Tensor,
        pos_ids: torch.Tensor,
        rnd_neg: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> Union[List[torch.Tensor], Dict[str, torch.Tensor]]:
        """

        :param query: Query/anchor embeddings of shape (N_q, D_q)
        :param pos: Embeddings of positives  of shape (N_q, D_i)
        :param rnd_neg: Embeddings of random negatives of shape (N_n, D_i). Optional when negative mode is IN_BATCH.
        :param query_ids: Ids of the queries of shape (N_q, )
        :param pos_ids: Ids of the positives od shape (N_q, )
        :param neg_ids: Ids of the random negatives of shape (N_n, ). Optional when negative mode is IN_BATCH.
        :param weights: Weights for each query of shape (N_q, )
        :return: Loss of the batch
        """

        estimated_counts = self._estimate_freq_counts(query_ids=query_ids, pos_ids=pos_ids, neg_ids=neg_ids)

        all_reduce_pos, mask = self._sample_in_batch_negatives(pos=pos)

        estimated_counts = self._mask_estimated_counts(mask, estimated_counts)

        self._estimated_freq_counts = estimated_counts
        self._num_in_batch_negs = all_reduce_pos.size(0)
        self._num_pairs = query.size(0)

        if self.neg_mode not in (SoftmaxNegativeMode.IN_BATCH,):
            all_reduce_rnd_neg = (
                torch.cat(AllGatherWithGrad.apply(rnd_neg), dim=0) if self._synchronize_negs else rnd_neg
            )
        else:
            all_reduce_rnd_neg = None

        if self.inverse_pos_freq_weight > 0.0:
            pos_cts, total_cts = self.in_batch_neg_ct.freq_est(pos_ids)
            pos_prob = pos_cts.view(-1) / total_cts
            w = torch.exp(self.inverse_pos_freq_weight * (1 - pos_prob / pos_prob.median()))
            weights *= w

        return self.compute_similarity_loss(
            query=query,
            pos=pos,
            all_reduce_pos=all_reduce_pos,
            all_reduce_rnd_neg=all_reduce_rnd_neg,
            weights=weights,
            estimated_counts=estimated_counts,
        )

    @torch.jit.ignore
    def write_summary(self, summary_writer: SummaryWriter, epoch: int, iteration: int, total_iterations: int) -> None:
        """
        Some useful visualizations to store in tensorboard.
        """
        niter = calc_global_iteration(epoch, iteration, total_iterations)

        # scalars are cheaper to log
        if iteration % 100 == 0:
            root_tag = f"{self._summary_prefix}/stats".lstrip("/")
            summary_writer.add_scalar(f"{root_tag}/num_posnegs", self._num_in_batch_negs, global_step=niter)
            summary_writer.add_scalar(f"{root_tag}/num_pairs", self._num_pairs, global_step=niter)

        if iteration % 1000 == 0:
            for counter, name in [
                (self._estimated_freq_counts.in_batch_neg_cts, "in_batch_neg_probs"),
                (self._estimated_freq_counts.rnd_neg_cts, "rnd_neg_probs"),
            ]:
                if counter is not None:
                    summary_writer.add_histogram(
                        tag=f"{self._summary_prefix}/stats/{name}".lstrip("/"),
                        values=torch.exp(counter.estimated_log_prob()).cpu().view(-1).numpy(),
                        global_step=niter,
                    )
            if self._estimated_freq_counts.conditional_cts is not None:
                probs = torch.exp(self._estimated_freq_counts.conditional_cts.estimated_log_prob())
                summary_writer.add_histogram(
                    tag=f"{self._summary_prefix}/stats/p_given_q_probs".lstrip("/"),
                    values=probs.cpu().view(-1).numpy(),
                    global_step=niter,
                )

            summary_writer.add_scalar(
                tag=f"{self._summary_prefix}/debug_metrics/loss_temperature".lstrip("/"),
                scalar_value=self.temperature.item(),
                global_step=niter,
            )
            if self._estimated_freq_counts.in_batch_neg_cts is not None:
                value = self._estimated_freq_counts.in_batch_neg_cts.all_ids.unique(return_counts=True)[1].max().item()
                summary_writer.add_scalar(
                    tag=f"{self._summary_prefix}/debug_metrics/max_pos_freq".lstrip("/"),
                    scalar_value=value,
                    global_step=niter,
                )


class MultiDimSoftmaxCorrectionLossV2(SoftmaxCorrectionLoss):
    """Compute softmax correction loss for different embedding dimension sizes as well as weights.

    Unlike `MultiDimSoftmaxCorrectionLoss`, this one inherits the `SoftmaxCorrectionLoss` module, which would has the following
    advantages:
    - Cleaner implementation and easier to extend and modify
    - Allow different negative modes (in-batch, rand neg or combined)
    - Allow different correction strategy (configured by params like `enable_spc`, `apply_pos_spc`, `average_neg_logits`)
    - Allow the query size to be different across multiple nodes/gpus
    """

    def __init__(self, *, emb_dim_weights: Dict[int, float], **kwargs):
        super().__init__(**kwargs)
        self.emb_dim_weights = emb_dim_weights

    def extra_repr(self) -> str:
        extra_reprs = [
            super().extra_repr(),
            f'emb_dim_weights={",".join(f"{k}:{v}" for k, v in self.emb_dim_weights.items())}',
        ]
        return ", ".join(extra_reprs)

    def compute_similarity_loss(
        self,
        query: torch.Tensor,
        pos: torch.Tensor,
        all_reduce_pos: torch.Tensor,
        all_reduce_rnd_neg: Optional[torch.Tensor],
        weights: torch.Tensor,
        estimated_counts: EstimatedFreqCounts,
    ) -> Union[List[torch.Tensor], Dict[str, List[Tensor]]]:
        losses = {}

        def unpack_fn(x):
            return x.items() if self.return_dict else enumerate(x)

        for dim, wt in self.emb_dim_weights.items():
            # we skip when dim is 512 and pos pairs are from fixed tower with 256 size
            if dim <= pos.shape[1]:
                dim_losses = super().compute_similarity_loss(
                    query=F.normalize(query[:, :dim]),
                    pos=F.normalize(pos[:, :dim]),
                    all_reduce_pos=F.normalize(all_reduce_pos[:, :dim]),
                    all_reduce_rnd_neg=F.normalize(all_reduce_rnd_neg[:, :dim])
                    if all_reduce_rnd_neg is not None
                    else None,
                    weights=weights,
                    estimated_counts=estimated_counts,
                )
                for k, loss in unpack_fn(dim_losses):
                    losses[f"{dim}_{k}"] = loss * wt

        return losses if self.return_dict else list(losses.values())
