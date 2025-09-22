from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import cast
from typing_extensions import Protocol

import enum
import operator
from functools import reduce

import numpy as np
import sklearn.metrics as metrics
import torch

from interactrank.common.utils.utils import send_to_device
from interactrank.common.utils.metrics import EPSILON
from interactrank.common.utils.metrics import interpolate_pr_auc
from interactrank.common.utils.metrics import interpolate_roc_auc

if TYPE_CHECKING:
    from torch import Tensor

T = TypeVar("T", contravariant=True)
Result = TypeVar("Result", covariant=True)
PartialResult = TypeVar("PartialResult")


class Accumulator(Protocol[T, PartialResult, Result]):
    """
    Accumulates reduction result from a stream of items
    """

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize accumulator state if required. It is optional to implement this method.
        :param kwargs: Args for the initializer
        """
        pass

    def accept(self, item: T) -> None:
        """
        Accept an item and update the accumulator state
        :param item: Item to process
        """
        raise NotImplementedError

    def value(self) -> Result:
        """
        :return: Result of the accumulation so far
        """
        raise NotImplementedError

    def partial_value(self) -> PartialResult:
        """
        :return: Returns a representation of the accumulator state so that partial results from multiple accumulators
        can be combined to generate a final result
        """
        raise NotImplementedError

    def combine(self, partial_results: List[PartialResult]) -> Result:
        """
        :param partial_results: Partial results from different accumulators
        :return: Combined accumulation result after combining the different partial accumulations
        """
        ...

    def close(self) -> None:
        """
        Close the accumulator and release any resources
        """
        pass


InferenceResultAccumulator = Accumulator[Dict[str, Any], PartialResult, Result]

S = TypeVar("S")


class FoldAccumulator(Accumulator[S, S, S]):
    """
    An accumulator that performs a fold operation on the values passed to this accumulator.
    The operation can be customized and is simply a callable acting on two values that need to be accumulated over.
    """

    def __init__(self, operation: Callable[[S, S], S], initial_state: Optional[S] = None):
        self.state = initial_state
        self.operation = operation

    def accept(self, element: S) -> None:
        if self.state is None:
            self.state = element
        else:
            self.state = self.operation(self.state, element)

    def value(self) -> S:
        assert self.state is not None, "value() called for an empty accumulator"
        return self.state

    partial_value = value

    def combine(self, partial_results: List[S]) -> S:
        assert len(partial_results) > 0, "Passed in partial results needs to have at least one element to combine."
        return reduce(self.operation, partial_results[1:], partial_results[0])


def make_reduce_accumulator(operation: Callable[[S, S], S]) -> FoldAccumulator[S]:
    """Creates a reduce accumulator. A reduce accumulator doesn't need an initial state.

    Args:
        operation (Callable[[S, S], S]): [description]

    Returns:
        FoldAccumulator: [description]
    """
    return make_fold_accumulator(operation=operation)


def make_tensor_sum_accumulator(init_sum: Optional[torch.Tensor] = None) -> FoldAccumulator[torch.Tensor]:
    """Accumulator that operates on Tensors and performs a sum.

    Args:
        init_sum (Optional[torch.Tensor], optional): [description]. Defaults to None.

    Returns:
        FoldAccumulator[torch.Tensor]: [description]
    """
    if init_sum is not None:
        assert init_sum.requires_grad is False

    return FoldAccumulator[torch.Tensor](operator.iadd, init_sum)


def make_fold_accumulator(operation: Callable[[S, S], S], init_value: Optional[S] = None) -> FoldAccumulator[S]:
    """Creates a fold accumulator using an operation and a initial_value

    Args:
        operation (Callable[[S, S], S]): [description]:
        init_value (Optional[S], optional): [description]. Defaults to None.

    Returns:
        FoldAccumulator[S]: [description]
    """
    return FoldAccumulator[S](operation, init_value)


V = TypeVar("V")


class SumTensorAccumulator(InferenceResultAccumulator[torch.Tensor, torch.Tensor]):
    """
    Calculates sum for given values key
    """

    def __init__(self, values_key: str) -> None:
        self.sum = make_tensor_sum_accumulator()
        self.values_key = values_key

    def accept(self, inference_result: Dict[str, Any]) -> None:
        assert inference_result.get(self.values_key, None) is not None, (
            f"Expect inference_result[{self.values_key}] not None, but got None"
        )
        self.sum.accept(inference_result[self.values_key])

    def value(self) -> torch.Tensor:
        return self.sum.value()

    def partial_value(self) -> torch.Tensor:
        return self.sum.value()

    @staticmethod
    def combine(partial_results: List[torch.Tensor]) -> torch.Tensor:
        devices = set(tensor.device for tensor in partial_results)
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, torch.device("cuda"))
        return torch.stack(partial_results, dim=0).sum(dim=0)


class AccuracyAccumulator(InferenceResultAccumulator[Tuple[int, torch.Tensor], torch.Tensor]):
    """
    Calculates the accuracy of inference results. Supports both single and multi-head tasks.
    """

    def __init__(self, predictions_key: str = "predictions", labels_key: str = "labels"):
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.num_samples = make_tensor_sum_accumulator()
        self.num_correct = make_tensor_sum_accumulator()

    def accept(self, inference_result: Dict[str, Any]) -> None:
        labels = inference_result[self.labels_key]
        predictions = inference_result[self.predictions_key]
        self.num_samples.accept(torch.as_tensor(labels.shape[0]))
        self.num_correct.accept(torch.sum(torch.eq(labels, predictions), dim=0))

    def partial_value(self) -> Tuple[int, torch.Tensor]:
        num_samples = self.num_samples.value()
        num_correct = self.num_correct.value()

        return num_samples, num_correct

    def value(self) -> torch.Tensor:
        num_samples = self.num_samples.value()
        num_correct = self.num_correct.value()

        return torch.div(num_correct, num_samples)

    @staticmethod
    def combine(partial_results: List[Tuple[int, torch.Tensor]]) -> torch.Tensor:
        devices = set(t[1].device for t in partial_results)
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, torch.device("cuda"))
        total_samples = sum(x[0] for x in partial_results)
        total_corrects = torch.stack([x[1] for x in partial_results], dim=0).sum(dim=0)
        return torch.div(total_corrects, total_samples)


class WeightedAverageAccumulator(InferenceResultAccumulator[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]):
    """
    Calculates weighted average for given values & weights key

    Args:
        values_key (str): Values key to extract from inference dict result.
        weights_key (Optional[str]): Weights key to extract from inference dict result, will use 1.0 as default.

    Usage:
    >>> avg_loss = WeightedAverageAccumulator(values_key="loss")
    >>> avg_loss.accept({"loss": 100})
    >>> avg_loss.accept({"loss": 200})
    >>> avg_loss.value()
    tensor(150.)

    """

    def __init__(self, values_key: str, weights_key: Optional[str] = None) -> None:
        self.values_sum = make_tensor_sum_accumulator()
        self.weights_sum = make_tensor_sum_accumulator()
        self.values_key = values_key
        self.weights_key = weights_key

    def accept(self, inference_result: Dict[str, Any]):
        values = inference_result[self.values_key]
        if self.weights_key is None:
            weights = 1.0
        else:
            weights = inference_result.get(self.weights_key, 1.0)

        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, device=values.device)

        weights = torch.broadcast_to(weights, values.shape)

        self.values_sum.accept((values * weights).sum())
        self.weights_sum.accept(weights.sum())

    def accept_partial_value(self, partial_value: Tuple[torch.Tensor, torch.Tensor]) -> None:
        values, weights = partial_value

        self.values_sum.accept(values.sum())
        self.weights_sum.accept(weights.sum())

    def value(self) -> torch.Tensor:
        values = self.values_sum.value()
        weights = self.weights_sum.value()

        return values / weights

    def partial_value(self) -> Tuple[torch.Tensor, torch.Tensor]:
        values = self.values_sum.value()
        weights = self.weights_sum.value()

        return values, weights

    @staticmethod
    def combine(partial_results: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        devices = set(t[0].device for t in partial_results)
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, torch.device("cuda"))
        total_values = torch.sum(torch.stack([x[0] for x in partial_results]))
        total_weights = torch.sum(torch.stack([x[1] for x in partial_results]))
        return torch.div(total_values, total_weights)


class ClkDurationRMSEAccumulator(InferenceResultAccumulator[Dict[str, Any], torch.Tensor]):
    """
    Accumulator for calculating Mean Squared Error (MSE). Predictions is of shape (batch_size, len(bin)-1)
    is from ordinal regression. Specific transformation _transform is applied before accept.

    Args:
        predictions_key (str): The key to identify prediction values in the inference result.
        labels_key (str): The key to identify label values in the inference result.
    """

    def __init__(self, predictions_key: str, labels_key: str, weights_key: str):
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.weights_key = weights_key
        self.diff_square_sum = make_tensor_sum_accumulator()
        self.num_samples = make_tensor_sum_accumulator()

    # prediction is of shape (batch_size, len(bin)-1), transform to the size of batch_size, each element in [0, len(bin)-1]
    def _transform(self, predictions):
        return (predictions > 0.5).sum(dim=1)

    def accept(self, inference_result: Dict[str, Any]) -> None:
        predictions = inference_result[self.predictions_key]
        labels = inference_result[self.labels_key]
        weights = inference_result[self.weights_key]

        predictions = self._transform(predictions)

        self.diff_square_sum.accept((weights * (predictions - labels).pow(2)).sum())
        self.num_samples.accept(weights.sum())

    def accept_partial_value(self, partial_value: Dict[str, torch.Tensor]) -> None:
        self.diff_square_sum.accept(partial_value['diff_square_sum'])
        self.num_samples.accept(partial_value['nsamples'])

    def value(self) -> torch.Tensor:
        diff_square_sum = self.diff_square_sum.value()
        nsamples = self.num_samples.value()

        return torch.sqrt(diff_square_sum / nsamples)

    def partial_value(self) -> Dict[str, torch.Tensor]:
        return {'diff_square_sum': self.diff_square_sum.value(), 'nsamples': self.num_samples.value()}

    def combine(self, partial_results: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        devices = set(t.device for d in partial_results for t in d.values())
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, torch.device("cuda"))
        diff_square_sum = torch.sum(torch.stack([x['diff_square_sum'] for x in partial_results]))
        nsamples = torch.sum(torch.stack([x['nsamples'] for x in partial_results]))

        return torch.sqrt(diff_square_sum / nsamples)


class ClkDurationMAEAccumulator(WeightedAverageAccumulator):
    """
    Accumulator for calculating Binned Mean Absolute Error (MAE). Predictions is of shape (batch_size, len(bin)-1)
    is from ordinal regression. Specific transformation _transform is applied before accept.

    Args:
        predictions_key (str): The key to identify prediction values in the inference result.
        labels_key (str): The key to identify label values in the inference result.
    """

    def __init__(self, predictions_key: str, labels_key: str, weights_key: str):
        super().__init__(values_key=predictions_key, weights_key=weights_key)
        self.labels_key = labels_key

    # prediction is of shape (batch_size, len(bin)-1), transform to the size of batch_size, each element in [0, len(bin)-1]
    def _transform(self, predictions):
        return (predictions > 0.5).sum(dim=1)

    def accept(self, inference_result: Dict[str, Any]) -> None:
        values = inference_result[self.values_key]
        labels = inference_result[self.labels_key]
        weights = inference_result[self.weights_key]

        values = self._transform(values)

        self.values_sum.accept((weights * torch.abs(values - labels)).sum())
        self.weights_sum.accept(weights.sum())


class WeightedSumAccumulator(InferenceResultAccumulator[torch.Tensor, torch.Tensor]):
    """
    Calculates weighted sum for given values & weights key

    Args:
        values_key (str): Values key to extract from inference dict result.
        weights_key (Optional[str]): Weights key to extract from inference dict result, will use 1.0 as default.

    Usage:
    >>> sum_loss = WeightedSumAccumulator(values_key="loss")
    >>> sum_loss.accept({"loss": 1})
    >>> sum_loss.accept({"loss": 2})
    >>> sum_loss.value()
    tensor(3)

    """

    def __init__(self, values_key: str, weights_key: Optional[str] = None) -> None:
        self.values_key = values_key
        self.weights_key = weights_key
        self.values_sum = make_tensor_sum_accumulator()

    def accept(self, inference_result: Dict[str, Any]):
        values = inference_result[self.values_key]
        if self.weights_key is None:
            weights = 1.0
        else:
            weights = inference_result.get(self.weights_key, 1.0)

        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values)
        if not isinstance(weights, torch.Tensor):
            weights = torch.as_tensor(weights, device=values.device)

        weights = torch.broadcast_to(weights, values.shape)

        self.values_sum.accept((values * weights).sum())

    def accept_partial_value(self, partial_value: torch.Tensor) -> None:
        self.values_sum.accept(partial_value.sum())

    def value(self) -> torch.Tensor:
        return self.values_sum.value()

    def partial_value(self) -> torch.Tensor:
        return self.values_sum.value()

    @staticmethod
    def combine(partial_results: List[torch.Tensor]) -> torch.Tensor:
        devices = set(t.device for t in partial_results)
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, torch.device("cuda"))
        total_values = torch.sum(torch.stack([x for x in partial_results]))
        return total_values


class ClkDurationPredictionLabelDistAccumulator(InferenceResultAccumulator[Dict[str, Any], torch.Tensor]):
    """
    Accumulator for calculating prediction rate and label rate in each bin. Predictions is of shape (batch_size, len(bin)-1)
    is from ordinal regression. Specific transformation _transform is applied before accept.

    Args:
        predictions_key (str): The key to identify prediction values in the inference result.
        labels_key (str): The key to identify label values in the inference result.
    """

    def __init__(self, predictions_key: str, labels_key: str, weights_key: str, num_bins: int):
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.weights_key = weights_key
        self.device: Optional[torch.device] = None
        self.num_bins = num_bins
        # int prediction, each element lies in [0, ..., num_bins-1]
        self.prediction_dist = make_tensor_sum_accumulator(torch.zeros(self.num_bins))
        # int label, each element lies in [0, ..., num_bins-1]
        self.label_dist = make_tensor_sum_accumulator(torch.zeros(self.num_bins))
        # raw prediction, each element x_i is P(1 appears >= index i+1 )
        self.prediction_agg_dist = make_tensor_sum_accumulator(torch.zeros(self.num_bins - 1))
        # raw label, e.g., (1, 1, 1, 0)
        self.label_agg_dist = make_tensor_sum_accumulator(torch.zeros(self.num_bins - 1))

    def _init_device(self, device: torch.device):
        self.device = device
        self.prediction_dist.state = cast(torch.Tensor, self.prediction_dist.state).to(self.device)
        self.label_dist.state = cast(torch.Tensor, self.label_dist.state).to(self.device)
        self.prediction_agg_dist.state = cast(torch.Tensor, self.prediction_agg_dist.state).to(self.device)
        self.label_agg_dist.state = cast(torch.Tensor, self.label_agg_dist.state).to(self.device)

    # prediction is of shape (batch_size, len(bin)-1), transform to the size of batch_size, each element in [0, len(bin)-1]
    def _prediction_cum2int_transform(self, predictions):
        return (predictions > 0.5).sum(dim=1)

    # transform label in [0, ..., num_bins-1] to aggregated multihot encoding
    # e.x. 3 -> [0,0,0,1,0] -> [1,1,1,1,0] -> [1,1,1,0]
    def _label_int2cum_transform(self, label):
        m = label.size(0)
        one_hot_label = torch.zeros((m, self.num_bins), device=self.device)
        one_hot_label[torch.arange(m), label.to(torch.int)] = 1.0
        one_hot_label = torch.cummax(one_hot_label.flip(1), dim=1)[0].flip(1)[:, 1:].float()
        return one_hot_label

    def accept(self, inference_result: Dict[str, torch.Tensor]):
        cum_raw_predictions = inference_result[self.predictions_key]
        int_labels = inference_result[self.labels_key]
        weights = inference_result[self.weights_key]

        if not self.device:
            self._init_device(cum_raw_predictions.device)

        int_predictions = self._prediction_cum2int_transform(cum_raw_predictions).to(self.device)
        cum_labels = self._label_int2cum_transform(int_labels).to(self.device)

        # cast to 2-D for following broadcast operation
        int_predictions = int_predictions.reshape(-1, 1)
        int_labels = int_labels.reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        prediction_d = torch.zeros(self.num_bins, device=self.device)
        label_d = torch.zeros(self.num_bins, device=self.device)

        prediction_agg_d = (cum_raw_predictions * weights).sum(dim=0)
        label_agg_d = (cum_labels * weights).sum(dim=0)

        for i in range(self.num_bins):
            prediction_d[i] = ((int_predictions == i) & (weights != 0)).sum()
            label_d[i] = ((int_labels == i) & (weights != 0)).sum()

        self.prediction_dist.accept(prediction_d)
        self.label_dist.accept(label_d)
        self.prediction_agg_dist.accept(prediction_agg_d)
        self.label_agg_dist.accept(label_agg_d)

    def accept_partial_value(self, partial_value: Dict[str, torch.Tensor]) -> None:
        self.prediction_dist.accept(partial_value["prediction_dist"])
        self.label_dist.accept(partial_value["label_dist"])

    def _metrics_generation(self, d):
        results = {}
        results['calibration_agg_dist'] = d['prediction_agg_dist'] / (d['label_agg_dist'] + EPSILON)
        results['prediction_dist'] = d['prediction_dist']
        results['label_dist'] = d['label_dist']
        results['prediction_agg_dist'] = d['prediction_agg_dist']
        results['label_agg_dist'] = d['label_agg_dist']
        return results

    def value(self) -> Dict[str, torch.Tensor]:
        return self._metrics_generation(self.partial_value())

    def partial_value(self) -> Dict[str, torch.Tensor]:
        prediction_dist = self.prediction_dist.value()
        label_dist = self.label_dist.value()
        prediction_agg_dist = self.prediction_agg_dist.value()
        label_agg_dist = self.label_agg_dist.value()
        return {
            "prediction_dist": prediction_dist,
            "label_dist": label_dist,
            "prediction_agg_dist": prediction_agg_dist,
            "label_agg_dist": label_agg_dist,
        }

    def combine(self, partial_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        res = {}
        devices = set(tensor.device for d in partial_results for tensor in d.values())
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, self.device)
        for key in ("prediction_dist", "label_dist", "prediction_agg_dist", "label_agg_dist"):
            res[key] = torch.stack([d[key] for d in partial_results], dim=0).sum(dim=0)

        return self._metrics_generation(res)


class RecallAtFPREstimator(InferenceResultAccumulator[Dict[str, torch.Tensor], torch.Tensor]):
    def __init__(self, predictions_key: str, labels_key: str, fpr: float = 0.01, truncated_roc: bool = False):
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.fpr = fpr
        self.predictions = []
        self.labels = []
        self.truncated_roc = truncated_roc

    def accept(self, inference_result: Dict[str, torch.Tensor]) -> None:
        self.predictions.append(inference_result[self.predictions_key])
        self.labels.append(inference_result[self.labels_key])

    def partial_value(self) -> Dict[str, Tensor]:
        return {"preds": torch.cat(self.predictions), "labels": torch.cat(self.labels)}

    def _calc(self, preds, labels) -> Tensor:
        fpr, tpr, thresholds = metrics.roc_curve(y_true=labels, y_score=preds)

        for i in range(len(fpr)):
            if fpr[i] > self.fpr:
                threshold = thresholds[i - 1]
                if not self.truncated_roc:
                    return torch.tensor(metrics.recall_score(y_true=labels, y_pred=preds >= threshold))
                else:
                    if i > 2:
                        return torch.tensor(metrics.auc(x=fpr[: i - 1], y=tpr[: i - 1]) / self.fpr)
                    else:
                        return torch.tensor(0)
        return torch.tensor(1)

    def value(self) -> Tensor:
        val = self.partial_value()
        return self._calc(val["preds"].cpu().numpy(), val["labels"].cpu().numpy())

    def combine(self, partial_values: List[Dict[str, Tensor]]) -> Tensor:
        preds = torch.cat([x["preds"].cpu() for x in partial_values]).numpy()
        labels = torch.cat([x["labels"].cpu() for x in partial_values]).numpy()
        return self._calc(preds, labels)


class ClkDurationPrecisionRecallAccumulator(
    InferenceResultAccumulator[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
):
    def __init__(self, predictions_key: str, labels_key: str, weights_key: str, num_bins: int):
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.weights_key = weights_key
        self.num_bins = num_bins
        self.device: Optional[torch.device] = None
        self.TPs = make_tensor_sum_accumulator(torch.zeros(self.num_bins))
        self.TNs = make_tensor_sum_accumulator(torch.zeros(self.num_bins))
        self.FPs = make_tensor_sum_accumulator(torch.zeros(self.num_bins))
        self.FNs = make_tensor_sum_accumulator(torch.zeros(self.num_bins))

    def _init_device(self, device: torch.device):
        self.device = device
        self.TPs.state = cast(torch.Tensor, self.TPs.state).to(self.device)
        self.TNs.state = cast(torch.Tensor, self.TNs.state).to(self.device)
        self.FPs.state = cast(torch.Tensor, self.FPs.state).to(self.device)
        self.FNs.state = cast(torch.Tensor, self.FNs.state).to(self.device)

    # prediction is of shape (batch_size, len(bin)-1), transform to the size of batch_size, each element in [0, len(bin)-1]
    def _transform(self, predictions):
        return (predictions > 0.5).sum(dim=1)

    def accept(self, inference_result: Dict[str, torch.Tensor]):
        predictions = inference_result[self.predictions_key]
        labels = inference_result[self.labels_key]

        if not self.device:
            self._init_device(predictions.device)

        predictions = self._transform(predictions)

        weights: torch.Tensor = inference_result.get(self.weights_key, torch.ones_like(predictions).to(self.device))

        assert len(predictions.shape) == len(labels.shape) == len(weights.shape) == 1, (
            f"Please make sure `predictions ({predictions.shape})` `labels ({labels.shape})` `weights ({weights.shape})` provided are all in 1-D array format"
        )
        # cast to 2-D for following broadcast operation
        predictions = predictions.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        weights = weights.reshape(-1, 1)
        tp = torch.zeros(self.num_bins, device=self.device)
        fp = torch.zeros(self.num_bins, device=self.device)
        fn = torch.zeros(self.num_bins, device=self.device)
        tn = torch.zeros(self.num_bins, device=self.device)

        for i in range(self.num_bins):
            tp[i] = ((predictions == i) & (labels == i) & (weights != 0)).sum()
            fp[i] = ((predictions == i) & (labels != i) & (weights != 0)).sum()
            fn[i] = ((predictions != i) & (labels == i) & (weights != 0)).sum()
            tn[i] = ((predictions != i) & (labels != i) & (weights != 0)).sum()

        self.TPs.accept(tp)
        self.TNs.accept(tn)
        self.FPs.accept(fp)
        self.FNs.accept(fn)

    def accept_partial_value(self, partial_value: Dict[str, torch.Tensor]) -> None:
        self.TPs.accept(partial_value["TPs"])
        self.TNs.accept(partial_value["TNs"])
        self.FPs.accept(partial_value["FPs"])
        self.FNs.accept(partial_value["FNs"])

    def metrics_from_confusion_matrix(self, tps, tns, fps, fns):
        d = {}
        d['precision'] = tps / (tps + fps + EPSILON)
        d['recall'] = tps / (tps + fns + EPSILON)
        return d

    def value(self) -> Dict[str, torch.Tensor]:
        return self.metrics_from_confusion_matrix(
            tps=self.TPs.value(), tns=self.TNs.value(), fps=self.FPs.value(), fns=self.FNs.value()
        )

    def partial_value(self) -> Dict[str, torch.Tensor]:
        tps = self.TPs.value()
        tns = self.TNs.value()
        fps = self.FPs.value()
        fns = self.FNs.value()
        return {
            "TPs": tps,
            "TNs": tns,
            "FPs": fps,
            "FNs": fns,
        }

    def combine(self, partial_results: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        res = {}
        devices = set(tensor.device for d in partial_results for tensor in d.values())
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, self.device)
        for key in ("TPs", "TNs", "FPs", "FNs"):
            res[key] = torch.stack([d[key] for d in partial_results], dim=0).sum(dim=0)

        return self.metrics_from_confusion_matrix(res["TPs"], res["TNs"], res["FPs"], res["FNs"])


class BinnedConfusionMatrixAccumulatorBase(InferenceResultAccumulator[Dict[str, torch.Tensor], Result]):
    """
    Base class for accumulators that count true pos/true neg/false pos/false neg, then aggregate those counts
    somehow
    """

    def __init__(
        self,
        predictions_key: str,
        labels_key: str,
        weights_key: Optional[str] = None,
        num_thresholds: int = 101,
    ) -> None:
        self.predictions_key = predictions_key
        self.labels_key = labels_key
        self.weights_key = weights_key
        self.device: Optional[torch.device] = None

        self.num_thresholds = num_thresholds

        thresholds = [(i + 1) * 1.0 / (num_thresholds - 1) for i in range(num_thresholds - 2)]
        thresholds = [0.0 - EPSILON] + thresholds + [1.0 + EPSILON]
        self.thresholds = torch.tensor(thresholds)

        self.TPs = make_tensor_sum_accumulator(torch.zeros(num_thresholds))
        self.TNs = make_tensor_sum_accumulator(torch.zeros(num_thresholds))
        self.FPs = make_tensor_sum_accumulator(torch.zeros(num_thresholds))
        self.FNs = make_tensor_sum_accumulator(torch.zeros(num_thresholds))

    def _init_device(self, device: torch.device):
        self.device = device
        self.thresholds = self.thresholds.to(self.device)
        self.TPs.state = cast(torch.Tensor, self.TPs.state).to(self.device)
        self.TNs.state = cast(torch.Tensor, self.TNs.state).to(self.device)
        self.FPs.state = cast(torch.Tensor, self.FPs.state).to(self.device)
        self.FNs.state = cast(torch.Tensor, self.FNs.state).to(self.device)

    def accept(self, inference_result: Dict[str, torch.Tensor]):
        predictions: torch.Tensor = inference_result[self.predictions_key]
        labels: torch.Tensor = inference_result[self.labels_key]

        if not self.device:
            self._init_device(predictions.device)

        if self.weights_key is None:
            weights: torch.Tensor = torch.ones_like(predictions).to(self.device)
        else:
            weights: torch.Tensor = inference_result.get(self.weights_key, torch.ones_like(predictions).to(self.device))

        assert len(predictions.shape) == len(labels.shape) == len(weights.shape) == 1, (
            f"Please make sure `predictions ({predictions.shape})` `labels ({labels.shape})` `weights ({weights.shape})` provided are all in 1-D array format"
        )
        # cast to 2-D for following broadcast operation
        predictions = predictions.reshape(-1, 1)
        labels = labels.reshape(-1, 1)
        weights = weights.reshape(-1, 1)

        labels = torch.eq(labels, 1)
        predictions = (
            predictions >= self.thresholds
        )  # broadcast each prediction element with each threshold, and result in a cartesian product
        self.TPs.accept((weights * (labels & predictions)).sum(dim=0))
        self.TNs.accept((weights * ((~labels) & (~predictions))).sum(dim=0))
        self.FPs.accept((weights * ((~labels) & (predictions))).sum(dim=0))
        self.FNs.accept((weights * ((labels) & (~predictions))).sum(dim=0))

    def accept_partial_value(self, partial_value: Dict[str, torch.Tensor]) -> None:
        self.TPs.accept(partial_value["TPs"])
        self.TNs.accept(partial_value["TNs"])
        self.FPs.accept(partial_value["FPs"])
        self.FNs.accept(partial_value["FNs"])

    def value(self) -> Result:
        return self.metric_from_confusion_matrix(
            tps=self.TPs.value(), tns=self.TNs.value(), fps=self.FPs.value(), fns=self.FNs.value()
        )

    def partial_value(self) -> Dict[str, torch.Tensor]:
        tps = self.TPs.value()
        tns = self.TNs.value()
        fps = self.FPs.value()
        fns = self.FNs.value()
        return {
            "TPs": tps,
            "TNs": tns,
            "FPs": fps,
            "FNs": fns,
        }

    def metric_from_confusion_matrix(
        self, *, tps: torch.Tensor, tns: torch.Tensor, fps: torch.Tensor, fns: torch.Tensor
    ) -> Result:
        raise NotImplementedError

    def combine(self, partial_results: List[Dict[str, torch.Tensor]]) -> Result:
        res = {}
        devices = set(tensor.device for d in partial_results for tensor in d.values())
        if len(devices) > 1:
            partial_results = send_to_device(partial_results, self.device)
        for key in ("TPs", "TNs", "FPs", "FNs"):
            res[key] = torch.stack([d[key] for d in partial_results], dim=0).sum(dim=0)

        return self.metric_from_confusion_matrix(tps=res["TPs"], tns=res["TNs"], fps=res["FPs"], fns=res["FNs"])


class AUCAccumulator(BinnedConfusionMatrixAccumulatorBase[torch.Tensor]):
    """
    A non nn.module implementation of machine-learning/trainer/ppytorch/modules/metrics.py:BinnedAUC
    follow the pattern of accumulation across workers with `accept` `partial_value` `combine`

    Args:
        predictions_key (str): Predictions key to extract from inference dict result.
        labels_key (str): Labels key to extract from inference dict result.
        weights_key (Optional[str]): Weights key to extract from inference dict result, will use 1.0 as default.
        num_thresholds (int): The number of thresholds to use when discretizing the roc curve.
        curve (str): Specifies the name of the curve to be computed, 'ROC' [default] or 'PR' for the Precision-Recall-curve.

    Usage:
    >>> roc_auc_acc = AUCAccumulator(predictions_key="predictions", labels_key="labels", curve="ROC")
    >>> roc_auc_acc.accept({"predictions": torch.tensor([1,0,0,0,1]), "labels": torch.tensor([1,0,1,0,1])})
    >>> roc_auc_acc.value()
    tensor(0.8333)

    """

    class Curve(str, enum.Enum):
        ROC = "ROC"
        PR = "PR"

    def __init__(
        self,
        predictions_key: str,
        labels_key: str,
        weights_key: Optional[str] = None,
        num_thresholds: int = 101,
        curve: str = Curve.ROC,
    ) -> None:
        super().__init__(
            predictions_key=predictions_key,
            labels_key=labels_key,
            weights_key=weights_key,
            num_thresholds=num_thresholds,
        )

        assert curve in {"ROC", "PR"}, f"only support `ROC` & `PR` metrics, but got {curve}"
        self.curve = curve

    def metric_from_confusion_matrix(
        self, *, tps: torch.Tensor, tns: torch.Tensor, fps: torch.Tensor, fns: torch.Tensor
    ) -> torch.Tensor:
        if self.curve == "PR":
            return interpolate_pr_auc(tp=tps, fp=fps, fn=fns)
        return interpolate_roc_auc(tp=tps, tn=tns, fp=fps, fn=fns)


class RecallAtFixedPrecisionAccumulator(BinnedConfusionMatrixAccumulatorBase[torch.Tensor]):
    """
    Similar to AUCAccumulator, but computes recall at a given precision threshold (and does not support weights)
    """

    def __init__(self, predictions_key: str, labels_key: str, precision_threshold: float, num_thresholds: int = 1001):
        self.precision_threshold = precision_threshold
        super().__init__(
            predictions_key=predictions_key,
            labels_key=labels_key,
            weights_key=None,  # for now don't support weights
            num_thresholds=num_thresholds,
        )

    def metric_from_confusion_matrix(
        self, *, tps: torch.Tensor, tns: torch.Tensor, fps: torch.Tensor, fns: torch.Tensor
    ) -> torch.Tensor:
        recalls = tps / (tps + fns).clamp_min_(1e-6)
        precisions = tps / (tps + fps).clamp_min_(1e-6)

        recall_cands = recalls[precisions >= self.precision_threshold]

        if recall_cands.numel() == 0:
            return torch.zeros((), device=recall_cands.device)
        else:
            return recall_cands.max()


class OptimalThresholdForRecallAtFixedPrecisionAccumulator(BinnedConfusionMatrixAccumulatorBase[torch.Tensor]):
    """
    Similar to AUCAccumulator, but computes the optimal threshold for recall at a given precision threshold (and does not support weights)
    """

    def __init__(self, predictions_key: str, labels_key: str, precision_threshold: float, num_thresholds: int = 1001):
        self.precision_threshold = precision_threshold
        super().__init__(
            predictions_key=predictions_key,
            labels_key=labels_key,
            weights_key=None,  # for now don't support weights
            num_thresholds=num_thresholds,
        )

    def metric_from_confusion_matrix(
        self, *, tps: torch.Tensor, tns: torch.Tensor, fps: torch.Tensor, fns: torch.Tensor
    ) -> torch.Tensor:
        recalls = tps / (tps + fns).clamp_min_(1e-6)
        precisions = tps / (tps + fps).clamp_min_(1e-6)

        recall_cands = recalls[precisions >= self.precision_threshold]
        threshold_cands = self.thresholds[precisions >= self.precision_threshold]

        if recall_cands.numel() == 0:
            return torch.zeros((), device=recall_cands.device)
        else:
            return threshold_cands[recall_cands.argmax()]


class SegmentedBinnedAccumulator(InferenceResultAccumulator[Dict[str, torch.Tensor], Result]):
    """
    Computes metrics segmented by some boolean masks in the input dictionary. The masks to use are specified
    by mask_keys. the output is a map from mask_key -> result, where result is whatever the specified base_cls returns

    the input base_cls determines which accumulator to use for each segment, and it is constructed using passed
    in kwargs
    """

    accumulators: Dict[str, BinnedConfusionMatrixAccumulatorBase[Result]]

    def __init__(self, mask_keys: List[str], base_cls: Type[BinnedConfusionMatrixAccumulatorBase[Result]], **kwargs):
        self.accumulators = {k: base_cls(**kwargs) for k in mask_keys}

    def accept(self, item: Dict[str, torch.Tensor]) -> None:
        for k, acc in self.accumulators.items():
            mask = item[k]
            assert mask.dtype == torch.bool
            to_copy = [acc.predictions_key, acc.labels_key, acc.weights_key]
            acc.accept({k: item[k][mask] for k in to_copy if k is not None})

    def value(self) -> Dict[str, Result]:
        return {k: acc.value() for k, acc in self.accumulators.items()}

    def partial_value(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {k: acc.partial_value() for k, acc in self.accumulators.items()}

    def combine(self, partial_results: List[Dict[str, Dict[str, torch.Tensor]]]) -> Dict[str, Result]:
        out = {}
        for k, acc in self.accumulators.items():
            out[k] = acc.combine([pr[k] for pr in partial_results])
        return out


class SegmentedCountAccumulator(InferenceResultAccumulator[Dict[str, int], Dict[str, int]]):
    """
    Simple accumulator to count the number of values that are true for each of mask_keys. Returns a Dict[str, int]
    that counts the number of true values for each key in mask_keys
    """

    def __init__(self, mask_keys: List[str]):
        self.sums = {k: 0 for k in mask_keys}

    def accept(self, item: Dict[str, torch.Tensor]) -> None:
        for k in self.sums:
            mask = item[k]
            assert mask.dtype == torch.bool
            self.sums[k] += int(mask.long().sum().item())

    def value(self) -> Dict[str, int]:
        return self.sums

    def partial_value(self) -> Dict[str, int]:
        return self.sums

    def combine(self, partial_results: List[Dict[str, int]]) -> Dict[str, int]:
        out = {}

        for k in self.sums:
            out[k] = sum(r[k] for r in partial_results)
        return out


class SegmentedSumTensorAccumulator(SegmentedBinnedAccumulator[Dict[str, torch.Tensor]]):
    """
    A simple function to calculates sum for given values key segmented by mask_keys
    """

    accumulators: Dict[str, SumTensorAccumulator]

    def __init__(self, mask_keys: List[str], default_value: float = 0.0, **kwargs):
        self.accumulators = {k: SumTensorAccumulator(**kwargs) for k in mask_keys}
        self.default_value = default_value

    def accept(self, item: Dict[str, torch.Tensor]) -> None:
        for k, acc in self.accumulators.items():
            mask = item[k]
            assert mask.dtype == torch.bool
            item_copy = item[acc.values_key].clone()
            item_copy[~mask] = self.default_value
            acc.accept({acc.values_key: item_copy})


class FilteredBinnedAccumulator(InferenceResultAccumulator[Dict[str, torch.Tensor], Result]):
    """
    Computes metrics based on filtered inference results.

    The differences from SegmentedBinnedAccumulator are that
        1. For SegmentedBinnedAccumulator, multiple segments are defined by boolean masks, and only samples with mask=True are selected into each segment.
        For FilteredBinnedAccumulator, the target segment is defined by a filter, and the target value can be any value other than "True".
        2. The output of SegmentedBinnedAccumulator is a dict while the output of FilteredBinnedAccumulator is a Result just as any other base accumulator.

    Args:
        filter: A dictionary of (key, value) pairs to filter inference results, example: {"is_promoted": True}
        base_cls: The base class accumulator constructed by passed in kwargs, example: AUCAccumulator, AccuracyAccumulator, WeightedSumAccumulator, etc.
    """

    def __init__(self, filter: Dict[str, Any], base_cls, **kwargs):
        self.filter = filter
        self.accumulator = base_cls(**kwargs)
        self.non_empty_filtered_samples = False

    def accept(self, inference_result: Dict[str, torch.Tensor]):
        if isinstance(self.accumulator, WeightedSumAccumulator) or isinstance(
            self.accumulator, WeightedAverageAccumulator
        ):
            predictions = inference_result[self.accumulator.values_key]
        else:
            predictions = inference_result[self.accumulator.predictions_key]
        mask = torch.ones(predictions.shape[0], device=predictions.device).bool()
        for key, val in self.filter.items():
            if key not in inference_result:
                return
            mask &= inference_result[key] == val
            if not torch.any(mask):
                # return early if none of the results match
                return
        self.non_empty_filtered_samples = True
        if isinstance(self.accumulator, WeightedSumAccumulator) or isinstance(
            self.accumulator, WeightedAverageAccumulator
        ):
            to_copy = [self.accumulator.values_key]
        else:
            to_copy = [self.accumulator.predictions_key, self.accumulator.labels_key]
        if hasattr(self.accumulator, "weights_key") and self.accumulator.weights_key is not None:
            to_copy.append(self.accumulator.weights_key)
        # filter inference results based on mask
        self.accumulator.accept({k: inference_result[k][mask] for k in to_copy})

    def value(self) -> Optional[Result]:
        if self.non_empty_filtered_samples:
            return self.accumulator.value()

    def partial_value(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.non_empty_filtered_samples:
            return self.accumulator.partial_value()

    def combine(self, partial_results: List[Dict[str, torch.Tensor]]) -> Optional[Result]:
        if self.non_empty_filtered_samples:
            return self.accumulator.combine(partial_results)


class SimpleInferenceResultAccumulator(InferenceResultAccumulator):
    """
    A class that accumulates inference results in a simple manner.

    Args:
        value_keys (Set[str], optional): A set of keys to filter the inference results. When set not to none,
                                         only the keys present in this set will be accumulated. Defaults to None.
    """

    def __init__(self, value_keys: Optional[Set[str]] = None) -> None:
        """
        Initializes a new instance of the SimpleInferenceResultAccumulator class.

        Args:
            value_keys (Set[str], optional): A set of keys to filter the inference results. Only the keys present
                                             in this set will be accumulated. Defaults to None.
        """
        self.state = {}
        self.value_keys = value_keys

    def _update(self, store: Dict[str, Any], key: str, value: Any) -> None:
        if key not in store:
            store[key] = [] if not isinstance(value, dict) else {}
        if isinstance(value, list):
            store[key].extend(value)
        elif isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
            store[key].append(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                self._update(store[key], k, v)
        else:
            raise ValueError(f"Unexpected type in inference results: {type(value)} for {key}")

    def _merge(self, value: Any, merge_list=False) -> Any:
        current_device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(value, dict):
            return {k: self._merge(v) for k, v in value.items()}
        elif isinstance(value[0], np.ndarray):
            return np.concatenate(value, axis=0)
        elif isinstance(value[0], torch.Tensor):
            return torch.cat([v.to(current_device) for v in value], dim=0)
        elif isinstance(value[0], list) and merge_list:
            return sum(value, [])
        elif isinstance(value[0], dict) and merge_list:
            return {key: self._merge([v[key] for v in value]) for key in value[0]}
        return value

    def accept(self, inference_results: Dict[str, Any]) -> None:
        """
        Accepts the inference results and accumulates them.

        Args:
            inference_results (Dict[str, Any]): The inference results to accumulate.

        Raises:
            AssertionError: If the inference_results keys do not contain all the value_keys,
                            when value_keys is not None or empty.
        """
        if self.value_keys:
            assert self.value_keys.issubset(inference_results.keys()), (
                f"Expect inference_results keys {inference_results.keys()} contains {self.value_keys}"
            )
            inference_results = {k: inference_results[k] for k in self.value_keys}

        for key in inference_results:
            self._update(self.state, key, inference_results[key])

    def value(self) -> Dict[str, Any]:
        return {k: self._merge(v) for k, v in self.state.items()}

    partial_value = value
    accept_partial_value = accept

    def combine(self, partial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {key: self._merge([v[key] for v in partial_results], merge_list=True) for key in partial_results[0]}


class NDCGScoreAccumulator(InferenceResultAccumulator):
    """
    Accumulator for calculating the NDCG score given the predictions and scores.
    """

    def __init__(self, predictions_key: str, score_key: str, k=None, gains: str = "exponential"):
        self.predictions_key = predictions_key
        self.score_key = score_key
        self.k = k
        self.gains = gains
        self.predictions = []
        self.scores = []

    def accept(self, inference_result: Dict[str, Any]) -> None:
        self.predictions.append(inference_result[self.predictions_key])
        self.scores.append(inference_result[self.score_key])

    def value(self) -> torch.Tensor:
        val = self.partial_value()

        preds = val["preds"]
        scores = val["scores"]

        return torch.tensor(self._ndcg_score(scores, preds, self.k, self.gains))

    def partial_value(self) -> Dict[str, torch.Tensor]:
        return {"preds": torch.cat(self.predictions), "scores": torch.cat(self.scores)}

    def _dcg_score(self, scores: torch.Tensor, preds: torch.Tensor, k: int | None = None, gains: str = "exponential"):
        """Discounted cumulative gain (DCG) at rank k
        Parameters
        ----------
        scores : tensor, shape = [n_samples]
            Ground truth (true relevance labels).
        preds : tensor, shape = [n_samples]
            Predicted scores.
        k : int
            Rank.
        gains : str
            Whether gains should be "exponential" (default) or "linear".
        Returns
        -------
        DCG @k : float
        """
        order = torch.argsort(preds, descending=True)
        if k is None:
            scores = torch.take(scores, order)
        else:
            scores = torch.take(scores, order[:k])

        if gains == "exponential":
            gains = 2**scores - 1
        elif gains == "linear":
            gains = scores
        else:
            raise ValueError("Invalid gains option.")

        # highest rank is 1 so +2 instead of +1
        discounts = torch.log2(torch.arange(scores.shape[0]) + 2)
        return torch.sum(gains / discounts)

    def _ndcg_score(self, scores: torch.Tensor, preds: torch.Tensor, k: int | None = None, gains: str = "exponential"):
        best = self._dcg_score(scores, scores, k, gains)
        actual = self._dcg_score(scores, preds, k, gains)
        return actual / best

    def combine(self, partial_values: List[Dict[str, Tensor]]) -> torch.Tensor:
        preds = torch.cat([x["preds"] for x in partial_values])
        scores = torch.cat([x["scores"] for x in partial_values])

        return torch.tensor(self._ndcg_score(scores, preds, self.k, self.gains))
