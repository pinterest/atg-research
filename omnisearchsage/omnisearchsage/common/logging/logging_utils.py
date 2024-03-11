from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import Optional

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def write_scalar(summary_writer: Optional[SummaryWriter], metric_name: str, value, iteration: int):
    """write a scalar to tensorboard and the mlflow server if there's an active run"""
    if summary_writer:
        summary_writer.add_scalar(metric_name, value, iteration)


def write_scalars(
    summary_writer: Optional[SummaryWriter],
    metrics_map: Dict[str, float],
    iteration: int,
):
    """Write multiple scalars to tensorboard and the mlflow server if there's an active run
    For mlflow, this issues a single mlflow API call, and can reduce logging overhead substantially.
    Args:
        summary_writer:
        metrics_map: The metrics to emit, defined as:
            {str metric_name: float metric_value}
        iteration:
    """
    if summary_writer is not None:
        # I don't know how to batch metrics in tensorboard. Maybe add_scalars(main_tag="", scalar_dict_map)?
        for metric_name, value in metrics_map.items():
            summary_writer.add_scalar(metric_name, value, iteration)
