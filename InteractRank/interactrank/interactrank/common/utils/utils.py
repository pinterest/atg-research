from __future__ import annotations
from typing import Union
from typing import Callable
from typing import List
from typing import Dict
import numpy as np

import logging
import time
import errno
import hashlib
import dataclasses
import os

from dataclasses import dataclass
import pyarrow.dataset as ds
import pyarrow as pa
import torch
from torch import Tensor

ROC_AUC_STRIDE = 1024
ExampleType = Dict[str, Union[torch.Tensor, List[torch.Tensor]]]

LOG = logging.getLogger(__name__)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def hash_str(str_input):
    return hashlib.sha256(str_input.encode("utf-8")).hexdigest()

def time_and_log(func):
    def _wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        LOG.info(f"Execution time for {func.__name__}: {end_time - start_time} seconds")
        return result

    return _wrapper

def drop_negative_ids(emb_name_to_vocab: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {emb_name: [id_ for id_ in vocab if id_ >= 0] for emb_name, vocab in emb_name_to_vocab.items()}


@time_and_log
def compute_request_level_roc_auc_from_parquet(
    parquet_dir: str, device: str, request_id_field: str, prediciton_field: str, label_field: str
) -> Tensor:
    """
    Reads eval results from the specified parquet_dir and computes the request-level ROC AUC.
    The computation is done on the specified device.  See compute_request_level_roc_auc for an
    explanation of request-level ROC AUC.
    :param label_field: accumulator's label field
    :param prediciton_field: accumulator's prediction filed
    :param request_id_field: accumulator's request id field
    :param parquet_dir: the directory of the parquet files of eval results
    :param device: the torch device on which to do the computation
    :return: a scalar of the request-level ROC AUC
    """
    dataset = ds.dataset(parquet_dir, partitioning="hive")
    dataframe = dataset.to_table().to_pandas()
    request_ids = torch.tensor(dataframe[request_id_field].to_numpy())
    scores = torch.tensor(dataframe[prediciton_field].to_numpy())
    labels = torch.tensor(dataframe[label_field].to_numpy())
    return compute_request_level_roc_auc(request_ids, scores, labels, device)


def compute_request_level_roc_auc(request_ids, scores, labels: Tensor, device: str) -> Tensor:
    """
    Estimates the request-level ROC AUC using the given set of request IDs, prediction scores, and true labels.
    Request-level ROC AUC is computed by finding all pairs of positive/negative examples that share the same
    request ID.  The fraction of these pairs for which the positive example outscores the negative example is
    the request-level ROC AUC.

    The results of this routine are an estimate, because positive/negative examples pairs are only identified
    if they fall within a certain window (determined by ROC_AUC_STRIDE) after sorting by request ID.

    :param device: the torch device on which to do the computation
    :param request_ids: a tensor of the request IDs of each example
    :param scores: a tensor of the scores of each example (same size as request_ids)
    :param labels: a tensor of the labels of each example (same size as request_ids)
    :return: a scalar of the request-level ROC AUC
    """

    order = torch.argsort(request_ids.squeeze())
    request_ids = send_to_device(request_ids.squeeze()[order].unsqueeze(1), device)
    scores = send_to_device(scores.squeeze()[order].unsqueeze(1), device)
    labels = send_to_device(labels.squeeze()[order].unsqueeze(1), device)
    del order
    stride = ROC_AUC_STRIDE
    numerator = 0
    denominator = 0
    for i in range(0, request_ids.numel(), stride):
        if (i + 1) % 50000 == 0:
            progress = i / request_ids.numel() * 100
            LOG.info(f"request level roc progress: {progress}%.")
        # A mask of matrix entries that have the same user and query ID with the row corresponing to a
        # positive example and the column, a negative example.
        batch = slice(i, i + stride)
        # fmt: off
        same_request_pos_neg_pairs = (
            (request_ids[batch] == request_ids[batch].T)
            & (labels[batch] > labels[batch].T)
        )
        # fmt: on
        denominator += same_request_pos_neg_pairs.sum()
        # A mask of matrix entries where the row score is greater than the column score.
        pos_score_greater = scores[batch] > scores[batch].T
        numerator += (same_request_pos_neg_pairs & pos_score_greater).sum()
    return numerator / denominator

@torch.jit.script
def tensor_to_nonblocking(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)
def pin_and_copy_to_gpu(t: torch.Tensor, device: torch.device):
    t = t.pin_memory(device)
    return tensor_to_nonblocking(t, device)

def _apply_func(func: Callable[[torch.Tensor], torch.Tensor], data: Any, ignore_error: bool = False):
    """Recursively move tensors to the device"""
    if isinstance(data, pa.Array) or isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return func(data)

    if isinstance(data, list):
        if data and isinstance(data[0], (str, bytes)):
            return data
        return [_apply_func(func, v, ignore_error) for v in data]

    if isinstance(data, tuple):
        return tuple(_apply_func(func, v, ignore_error) for v in data)

    if isinstance(data, dict):
        return {k: _apply_func(func, v, ignore_error) for k, v in data.items()}

    if dataclasses.is_dataclass(data):
        args = {}
        for f in dataclasses.fields(data):
            args[f.name] = _apply_func(func, getattr(data, f.name), ignore_error)
        return type(data)(**args)

    if ignore_error:
        return data
    else:
        raise NotImplementedError(f"Unsupported type in tensor dict: {type(data)}")


def send_to_device(data, device: Union[torch.device, str], ignore_error: bool = False):
    """Recursively move tensors to the device"""
    if isinstance(device, str):
        device = torch.device(device)
    return _apply_func(lambda x: tensor_to_nonblocking(x, device), data=data, ignore_error=ignore_error)


def pin_memory(data, device: Union[torch.device, str], ignore_error: bool = False):
#     """Recursively move tensors to the device"""
     return _apply_func(lambda x: x.pin_memory(device), data=data, ignore_error=ignore_error)

def drop_negative_ids(emb_name_to_vocab: Dict[str, List[int]]) -> Dict[str, List[int]]:
    return {emb_name: [id_ for id_ in vocab if id_ >= 0] for emb_name, vocab in emb_name_to_vocab.items()}


@time_and_log
def compute_request_level_roc_auc_from_parquet(
    parquet_dir: str, device: str, request_id_field: str, prediciton_field: str, label_field: str
) -> Tensor:
    """
    Reads eval results from the specified parquet_dir and computes the request-level ROC AUC.
    The computation is done on the specified device.  See compute_request_level_roc_auc for an
    explanation of request-level ROC AUC.
    :param label_field: accumulator's label field
    :param prediciton_field: accumulator's prediction filed
    :param request_id_field: accumulator's request id field
    :param parquet_dir: the directory of the parquet files of eval results
    :param device: the torch device on which to do the computation
    :return: a scalar of the request-level ROC AUC
    """
    dataset = ds.dataset(parquet_dir, partitioning="hive")
    dataframe = dataset.to_table().to_pandas()
    request_ids = torch.tensor(dataframe[request_id_field].to_numpy())
    scores = torch.tensor(dataframe[prediciton_field].to_numpy())
    labels = torch.tensor(dataframe[label_field].to_numpy())
    return compute_request_level_roc_auc(request_ids, scores, labels, device)

@torch.jit.script
def tensor_to_nonblocking(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)


def pin_and_copy_to_gpu(t: torch.Tensor, device: torch.device):
    t = t.pin_memory(device)
    return tensor_to_nonblocking(t, device)
