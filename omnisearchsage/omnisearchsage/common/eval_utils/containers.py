from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import TypeVar
from typing import Union

import dataclasses
import os
from dataclasses import dataclass

import numpy as np
import torch
from omnisearchsage.common.utils.fs_utils import mkdir_p

if TYPE_CHECKING:
    from omnisearchsage.common.types import EntityType
    from omnisearchsage.data.sage_features import SageBatch
    from torch.utils.tensorboard import SummaryWriter

T = TypeVar("T", bound=Optional[Union[dict, list, np.ndarray]])


def permute_seq(v: T, perm) -> T:
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: permute_seq(v, perm) for k, v in v.items()}
    elif isinstance(v, list):
        if not v:
            return v
        return [v[j] for j in perm]
    elif isinstance(v, np.ndarray):
        return v[perm]
    elif isinstance(v, torch.Tensor):
        return v[perm]
    else:
        raise ValueError(f'Unable to permute type "{type(v)}"')


class Metric(NamedTuple):
    """
    Class to represent a metric.
    Args:
        metric_name: name of the Metric
        text_format_string: `text_format_string.format(metric.compute(positive_ranks))` will be printed wherever this
        metric is used
        compute: function that takes a numpy.ndarray of query ranks and, optionally a weight vector to use
            to compute this metric, and returns a scalar
    """

    metric_name: str
    text_format_string: str
    compute: Callable[[np.ndarray], float]

    def __repr__(self) -> str:
        return self.metric_name


class EvaluationMetadata(NamedTuple):
    """
    Class to pass around the metadata associated with current eval pairs, eval index, and iteration
    """

    summary_writer: Optional[SummaryWriter]
    prefix: str
    run_dir: str
    iteration: int

    metric_to_plot: Metric = None

    # batch size for eval
    batch_size: int = 128

    @property
    def absolute_prefix(self) -> str:
        path = os.path.join(self.run_dir, self.prefix)
        mkdir_p(path)
        return path


class EvaluationLoaderGroup(NamedTuple):
    """
    Wrapper for the entities needed to evaluate a model on a metric learning task.
    """

    # name of index. to be used for labeling metrics (can be anything, expected to be unique)
    index_name: str

    # for each index, we may wish to evaluate many sets of eval pairs. this is a {name: eval pair iterable} mapping
    query_positive_loaders: Dict[str, Iterable]

    # data loader for index features
    index_loader: Iterable[SageBatch]

    # batch size for eval computations
    batch_size: int = 128


@dataclass
class EvalEmbeddings:
    entity_type: EntityType
    keys: List[str]
    embeddings: Union[np.ndarray, torch.Tensor] = None

    def permute(self, perm) -> EvalEmbeddings:
        return dataclasses.replace(
            self,
            keys=permute_seq(self.keys, perm),
            embeddings=permute_seq(self.embeddings, perm),
        )

    def replace(self, **attrs) -> EvalEmbeddings:
        return dataclasses.replace(self, **attrs)
