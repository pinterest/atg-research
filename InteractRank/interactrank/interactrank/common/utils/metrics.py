from __future__ import annotations

from typing import NamedTuple
from typing import Optional
from typing import Union

import torch
import torch.nn as nn

EPSILON = 1.0e-7


class ConfusionMatrix(NamedTuple):
    tp: torch.Tensor
    tn: torch.Tensor
    fp: torch.Tensor
    fn: torch.Tensor


def div_no_nan(a, b):
    division = a / b
    division[b == 0] = 0
    return division


class SumMeter(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("value", torch.zeros(()))

    def reset_running_stats(self):
        self.value.zero_()

    def forward(self, value: Union[torch.Tensor, float]):
        """Updates the sum with a given tensor. This can be used to calculate number of
        positive labels / negative labels on top of a tensor that contains binary labels.
        Args:
            value: A tensor of observations (can also be a scalar value)
        """
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.value.device)

        self.value += value.sum()
        return self.value


class AverageMeter(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer("value", torch.zeros(()))
        self.register_buffer("weight", torch.zeros(()))

    def reset_running_stats(self):
        self.value.zero_()
        self.weight.zero_()

    def forward(self, value: Union[torch.Tensor, float], weight: Union[torch.Tensor, float] = 1.0):
        """Updates the average with.
        Args:
            value: A tensor of observations (can also be a scalar value)
            weight: The weight of each observation (automatically broadcasted
                to fit ``value``)
        """
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32, device=self.value.device)
        if not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=torch.float32, device=self.weight.device)

        weight = torch.broadcast_to(weight, value.shape)
        self.value += (value * weight).sum()
        self.weight += weight.sum()
        return self.value / self.weight


class BinnedAUC(nn.Module):
    """
    Computes the approximate AUC via a Riemann sum.

    Copied from implementation of https://www.tensorflow.org/api_docs/python/tf/compat/v1/metrics/auc
    translated to PyTorch.

    Supports `PR` curve or 'ROC' with `careful_interpolation`
    """

    def __init__(self, num_thresholds: int = 100, curve: str = "ROC", accumulated=True):
        super().__init__()

        self.curve = curve
        self.num_thresholds = num_thresholds
        self.accumulated = accumulated

        # linearly interpolate (num_thresholds - 2) thresholds in (0, 1).
        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        # Add an endpoint "threshold" below zero and above one for either threshold
        # method.
        thresholds = [0.0 - EPSILON] + thresholds + [1.0 + EPSILON]
        self.register_buffer("thresholds", torch.tensor(thresholds))

        for name in ("TPs", "TNs", "FPs", "FNs"):
            self.register_buffer(name, torch.zeros(num_thresholds, 1))

    def reset_running_stats(self):
        for name in ("TPs", "TNs", "FPs", "FNs"):
            getattr(self, name).zero_()

    def update_confusion_matrix(
        self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> ConfusionMatrix:
        """
        Extract the confusion matrix from the predictions and targets and update state if accumulating

        Args
            preds: (n_samples, 1) tensor
            targets: (n_samples, 1) tensor
            weights: (n_samples, 1) tensor

        Returns
            ConfusionMatrix
        """
        if weights is None:
            weights = torch.ones_like(preds)

        preds = preds.reshape(-1)
        targets = (targets == 1).reshape(-1)
        weights = weights.reshape(-1)

        # Iterate one threshold at a time to conserve memory
        bin_idx = torch.bucketize(preds, self.thresholds)

        tp = torch.zeros_like(self.TPs).squeeze_(1)
        tp.index_add_(dim=0, index=(bin_idx - 1).clamp_min_(0), source=(weights * targets).to(tp.dtype))
        tp = tp.flip(0).cumsum(0).flip(0)

        tn = torch.zeros_like(tp)
        tn.index_add_(dim=0, index=bin_idx, source=(weights * ~targets).to(tp.dtype))
        tn = tn.cumsum(0)

        fp = torch.zeros_like(tp)
        fp.index_add_(dim=0, index=(bin_idx - 1).clamp_min_(0), source=(weights * ~targets).to(tp.dtype))
        fp = fp.flip(0).cumsum(0).flip(0)

        fn = torch.zeros_like(tp)
        fn.index_add_(dim=0, index=bin_idx, source=(weights * targets).to(tp.dtype))
        fn = fn.cumsum(0)

        tp.unsqueeze_(1)
        tn.unsqueeze_(1)
        fp.unsqueeze_(1)
        fn.unsqueeze_(1)

        if self.accumulated:
            self.TPs += tp
            self.TNs += tn
            self.FPs += fp
            self.FNs += fn
            tp, tn, fp, fn = self.TPs, self.TNs, self.FPs, self.FNs

        return ConfusionMatrix(tp, tn, fp, fn)

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute the AUC score

        Args
            preds: (n_samples, 1) tensor
            targets: (n_samples, 1) tensor
            weights: (n_samples, 1) tensor
            only_interpolate: if True, only compute the interpolated AUC
        """
        # skip internal state updates when only logging
        confusion_matrix = self.update_confusion_matrix(preds, targets, weights)

        if self.curve == "PR":
            return interpolate_pr_auc(confusion_matrix.tp, confusion_matrix.fp, confusion_matrix.fn)
        elif self.curve == "ROC":
            return interpolate_roc_auc(
                confusion_matrix.tp, confusion_matrix.tn, confusion_matrix.fp, confusion_matrix.fn
            )
        else:
            raise NotImplementedError(f"Unsupported curve: {self.curve}")

    def pr_auc(self) -> torch.Tensor:
        """
        Compute the PR AUC score
        """
        assert self.accumulated, "To compute the PR AUC with this method, the accumulated flag must be set to True"
        return interpolate_pr_auc(self.TPs, self.FPs, self.FNs)

    def roc_auc(self) -> torch.Tensor:
        """
        Compute the ROC AUC score
        """
        assert self.accumulated, "To compute the ROC AUC with this method, the accumulated flag must be set to True"
        return interpolate_roc_auc(self.TPs, self.TNs, self.FPs, self.FNs)

    def extra_repr(self) -> str:
        return f"{self.curve}, num_thresholds={self.num_thresholds}"


def interpolate_roc_auc(tp, tn, fp, fn):
    # ROC calc for the rest
    x = fp / (fp + tn + EPSILON)  # false positive rate
    y = (tp + EPSILON) / (tp + fn + EPSILON)  # true positive rate
    roc = torch.sum((x[:-1] - x[1:]) * ((y[:-1] + y[1:]) / 2.0))
    return roc


def interpolate_pr_auc(tp, fp, fn):
    """Interpolation formula inspired by section 4 of (Davis et al., 2006).
    Note here we derive & use a closed formula not present in the paper
    - as follows:
    Modeling all of TP (true positive weight),
    FP (false positive weight) and their sum P = TP + FP (positive weight)
    as varying linearly within each interval [A, B] between successive
    thresholds, we get
      Precision = (TP_A + slope * (P - P_A)) / P
    with slope = dTP / dP = (TP_B - TP_A) / (P_B - P_A).
    The area within the interval is thus (slope / total_pos_weight) times
      int_A^B{Precision.dP} = int_A^B{(TP_A + slope * (P - P_A)) * dP / P}
      int_A^B{Precision.dP} = int_A^B{slope * dP + intercept * dP / P}
    where intercept = TP_A - slope * P_A = TP_B - slope * P_B, resulting in
      int_A^B{Precision.dP} = TP_B - TP_A + intercept * log(P_B / P_A)
    Bringing back the factor (slope / total_pos_weight) we'd put aside, we get
       slope * [dTP + intercept *  log(P_B / P_A)] / total_pos_weight
    where dTP == TP_B - TP_A.
    Note that when P_A == 0 the above calculation simplifies into
      int_A^B{Precision.dTP} = int_A^B{slope * dTP} = slope * (TP_B - TP_A)
    which is really equivalent to imputing constant precision throughout the
    first bucket having >0 true positives.
    Args:
      tp: true positive counts
      fp: false positive counts
      fn: false negative counts
    Returns:
      pr_auc: an approximation of the area under the P-R curve.
    References:
      The Relationship Between Precision-Recall and ROC Curves:
        [Davis et al., 2006](https://dl.acm.org/citation.cfm?id=1143874)
        ([pdf](https://www.biostat.wisc.edu/~page/rocpr.pdf))
    """
    dtp = tp[:-1] - tp[1:]
    p = tp + fp
    prec_slope = div_no_nan(dtp, torch.clamp(p[:-1] - p[1:], min=0))
    intercept = tp[1:] - (prec_slope * p[1:])
    safe_p_ratio = torch.where(
        (p[:-1] > 0) & (p[1:] > 0),
        div_no_nan(p[:-1], torch.clamp(p[1:], min=0)),
        torch.ones_like(p[1:]),
    )
    return torch.sum(
        div_no_nan(prec_slope * (dtp + intercept * torch.log(safe_p_ratio)), torch.clamp(tp[1:] + fn[1:], min=0))
    )
