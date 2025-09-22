from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from collections import OrderedDict

from interactrank.loggingcontent.logging_utils import write_scalars

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter


# Training stats
MLENV_TRAIN_METRICS_EXAMPLES_PER_SEC = "1_metrics_train/examples_per_sec"
MLENV_TRAIN_METRICS_TFLOP_PER_SEC = "1_metrics_train/tflop_per_sec"
MLENV_TRAIN_METRICS_BATCH_SEC = "1_metrics_train/batch_sec"
MLENV_TRAIN_METRICS_DATA_SEC = "1_metrics_train/data_sec"
MLENV_TRAIN_METRICS_FORWARD_SEC = "1_metrics_train/forward_sec"
MLENV_TRAIN_METRICS_BACKWARD_SEC = "1_metrics_train/backward_sec"


def calc_global_iteration(epoch: int, iteration: int, total_iterations: int) -> int:
    """
    Calculate current iteration number globally

    Args:
        epoch: current epoch
        iteration: current iteration within epoch
        total_iterations: total iterations of an epoch

    Returns:
        current iteration number globally
    """
    if total_iterations is None:
        return iteration

    return total_iterations * epoch + iteration


def write_summary(
    summary_writer: Optional[SummaryWriter],
    model: nn.Module,
    batch_time: float,
    data_time: float,
    forward_time: float,
    backward_time: float,
    total_loss: float,
    per_task_loss: Optional[Union[Sequence[float], float, Dict[str, float]]],
    learning_rate: Union[float, List[float]],
    epoch: int,
    iteration: int,
    total_iterations: Optional[int] = None,
    examples_per_sec: Optional[float] = None,
    task_names: Optional[Sequence[str]] = None,
    train_phase: int = 1,
    grad_norm: float = 0,
    use_mlflow_logging: bool = True,
    flop_per_sec: Optional[float] = None,
) -> None:
    """
    Shared common metrics we should always log in tensorboard

    Args:
        summary_writer: a tensorboardX.SummaryWriter instance
        model:  a EmbedModel
        batch_time:
        data_time:
        forward_time:
        backward_time:
        total_loss:
        grad_norm:
        per_task_loss:
        learning_rate: the current learning rate
        epoch: current epoch
        iteration: current iteration within epoch
        total_iterations: total iterations of an epoch
        examples_per_sec: training examples per second
        task_names: list of descriptive names for tasks
        train_phase: phase of training (for multiphase solvers)
        use_mlflow_logging: whether to use mlflow logging
        flop_per_sec: training FLOP per second
    """

    niter = calc_global_iteration(epoch, iteration, total_iterations)

    metrics_to_emit: OrderedDict[str, float] = OrderedDict()

    # Training stats
    if examples_per_sec:
        metrics_to_emit[MLENV_TRAIN_METRICS_EXAMPLES_PER_SEC] = examples_per_sec

    if flop_per_sec:
        metrics_to_emit[MLENV_TRAIN_METRICS_TFLOP_PER_SEC] = flop_per_sec / 1e12

    metrics_to_emit[MLENV_TRAIN_METRICS_BATCH_SEC] = batch_time
    metrics_to_emit[MLENV_TRAIN_METRICS_DATA_SEC] = data_time
    metrics_to_emit[MLENV_TRAIN_METRICS_FORWARD_SEC] = forward_time
    metrics_to_emit[MLENV_TRAIN_METRICS_BACKWARD_SEC] = backward_time
    learning_rate = [learning_rate] if isinstance(learning_rate, float) else learning_rate
    if len(learning_rate) == 1:
        metrics_to_emit[f"phase_{train_phase}/learning_rate"] = learning_rate[0]
    else:
        metrics_to_emit.update({f"phase_{train_phase}/learning_rate_{i}": lr for i, lr in enumerate(learning_rate)})
    metrics_to_emit[f"phase_{train_phase}/all_loss"] = total_loss
    metrics_to_emit[f"phase_{train_phase}/grad_norm"] = grad_norm
    metrics_to_emit["iteration"] = niter

    if isinstance(per_task_loss, list):
        task_names = task_names or range(len(per_task_loss))
        metrics_to_emit.update(
            {
                f"phase_{train_phase}/loss_{task_name}": task_loss
                for task_name, task_loss in zip(task_names, per_task_loss)
            }
        )
    if isinstance(per_task_loss, dict):
        metrics_to_emit.update(
            {f"phase_{train_phase}/loss_{task_name}": task_loss for task_name, task_loss in per_task_loss.items()}
        )

    # write_scalars() writes to both mlflow and summary_writer in a single batched request
    write_scalars(summary_writer, metrics_to_emit, niter)

    # Model stats to log
    if hasattr(model, "write_summary"):
        model.write_summary(summary_writer, epoch, iteration, total_iterations)
    elif hasattr(model, "module") and hasattr(model.module, "write_summary"):
        model.module.write_summary(summary_writer, epoch, iteration, total_iterations)
