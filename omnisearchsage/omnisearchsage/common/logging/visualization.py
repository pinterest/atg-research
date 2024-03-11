from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from collections import OrderedDict

import torch
from omnisearchsage.common.logging.device import SystemMetric
from omnisearchsage.common.logging.device import byte_to_gib
from omnisearchsage.common.logging.device import byte_to_mib
from omnisearchsage.common.logging.logging_utils import write_scalars

if TYPE_CHECKING:
    from torch import nn
    from torch.utils.tensorboard import SummaryWriter


# GPU metrics
GPU_METRICS_UTILIZATION = "5_metrics_gpu/utilization"
GPU_METRICS_MEMORY_GIB = "5_metrics_gpu/memory_GiB"
GPU_METRICS_MEMORY_ALLOCATED_GIB = "5_metrics_gpu/memory_allocated_GiB"
GPU_METRICS_MEMORY_MAX_ALLOCATED_GIB = "5_metrics_gpu/memory_max_allocated_GiB"
GPU_METRICS_MEMORY_RESERVED_GIB = "5_metrics_gpu/memory_reserved_GiB"
GPU_METRICS_MEMORY_MAX_RESERVED_GIB = "5_metrics_gpu/memory_max_reserved_GiB"

# CPU metrics
CPU_METRICS_PERCENT_AVAILABLE_MEM = "4_metrics_cpu/percent_available_mem"
CPU_METRICS_AVAILABLE_MEM_GIB = "4_metrics_cpu/available_mem_GiB"
CPU_METRICS_TOTAL_MEM_GIB = "4_metrics_cpu/total_mem_GiB"

# Network metrics
NETWORK_METRICS_BYTES_SENT_GIBPS = "3_metrics_network/bytes_sent_Gibps"
NETWORK_METRICS_BYTES_RECV_GIBPS = "3_metrics_network/bytes_recv_Gibps"
NETWORK_METRICS_BYTES_SENT_PER_SEC = "3_metrics_network/packages_sent_per_sec"
NETWORK_METRICS_BYTES_RECV_PER_SEC = "3_metrics_network/packages_recv_per_sec"

# Disk metrics
DISK_METRICS_NUM_READS_PER_SEC = "2_metrics_disk/num_reads_per_sec"
DISK_METRICS_NUM_WRITES_PER_SEC = "2_metrics_disk/num_writes_per_sec"
DISK_METRICS_NUM_READ_BYTES_MIBPS = "2_metrics_disk/read_bytes_MiBps"
DISK_METRICS_NUM_WRITE_BYTES_MIBPS = "2_metrics_disk/write_bytes_MiBps"

# Training stats
TRAIN_METRICS_EXAMPLES_PER_SEC = "1_metrics_train/examples_per_sec"
TRAIN_METRICS_BATCH_SEC = "1_metrics_train/batch_sec"
TRAIN_METRICS_DATA_SEC = "1_metrics_train/data_sec"
TRAIN_METRICS_FORWARD_SEC = "1_metrics_train/forward_sec"
TRAIN_METRICS_BACKWARD_SEC = "1_metrics_train/backward_sec"


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
    system_metrics: Optional[SystemMetric] = None,
    task_names: Optional[Sequence[str]] = None,
    train_phase: int = 1,
    grad_norm: float = 0,
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
        system_metrics: Object
        task_names: list of descriptive names for tasks
        train_phase: phase of training (for multiphase solvers)
    """
    system_metrics = system_metrics or SystemMetric()

    niter = calc_global_iteration(epoch, iteration, total_iterations)

    metrics_to_emit: OrderedDict[str, float] = get_system_metrics_to_emit(system_metrics)

    # Training stats
    if examples_per_sec:
        metrics_to_emit[TRAIN_METRICS_EXAMPLES_PER_SEC] = examples_per_sec

    metrics_to_emit[TRAIN_METRICS_BATCH_SEC] = batch_time
    metrics_to_emit[TRAIN_METRICS_DATA_SEC] = data_time
    metrics_to_emit[TRAIN_METRICS_FORWARD_SEC] = forward_time
    metrics_to_emit[TRAIN_METRICS_BACKWARD_SEC] = backward_time
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

    write_scalars(summary_writer, metrics_to_emit, niter)

    # Model stats to log
    if hasattr(model, "write_summary"):
        model.write_summary(summary_writer, epoch, iteration, total_iterations)
    elif hasattr(model, "module") and hasattr(model.module, "write_summary"):
        model.module.write_summary(summary_writer, epoch, iteration, total_iterations)


def get_system_metrics_to_emit(system_metrics: SystemMetric) -> OrderedDict[str, float]:
    """
    Shared system metrics to log in tensorboard

    Args:
        system_metrics: Object
    """
    metrics_to_emit = OrderedDict()
    if not system_metrics:
        return metrics_to_emit

    if torch.cuda.is_available():
        # GPU metrics
        gpu_utilization, gpu_memory = system_metrics.get_gpu_metrics(torch.cuda.current_device())
        metrics_to_emit[GPU_METRICS_UTILIZATION] = gpu_utilization
        metrics_to_emit[GPU_METRICS_MEMORY_GIB] = gpu_memory
        metrics_to_emit[GPU_METRICS_MEMORY_ALLOCATED_GIB] = byte_to_gib(torch.cuda.memory_allocated())
        metrics_to_emit[GPU_METRICS_MEMORY_MAX_ALLOCATED_GIB] = byte_to_gib(torch.cuda.max_memory_allocated())
        metrics_to_emit[GPU_METRICS_MEMORY_RESERVED_GIB] = byte_to_gib(torch.cuda.memory_reserved())
        metrics_to_emit[GPU_METRICS_MEMORY_MAX_RESERVED_GIB] = byte_to_gib(torch.cuda.max_memory_reserved())

    # CPU metrics
    available, total_memory, percentage_used = system_metrics.get_cpu_metrics()
    if available < 5.0:
        print("[Warning] Running low on available memory")

    metrics_to_emit[CPU_METRICS_PERCENT_AVAILABLE_MEM] = 100.0 - percentage_used
    metrics_to_emit[CPU_METRICS_AVAILABLE_MEM_GIB] = available
    metrics_to_emit[CPU_METRICS_TOTAL_MEM_GIB] = total_memory

    # Network metrics
    bytes_sent, bytes_recv, num_packages_sent, num_packages_recv = system_metrics.get_global_network_metrics()
    metrics_to_emit[NETWORK_METRICS_BYTES_SENT_GIBPS] = byte_to_gib(bytes_sent) * 8
    metrics_to_emit[NETWORK_METRICS_BYTES_RECV_GIBPS] = byte_to_gib(bytes_recv) * 8
    metrics_to_emit[NETWORK_METRICS_BYTES_SENT_PER_SEC] = num_packages_sent
    metrics_to_emit[NETWORK_METRICS_BYTES_RECV_PER_SEC] = num_packages_recv

    # Disk metrics
    read_count, write_count, read_bytes, write_bytes = system_metrics.get_global_disk_metrics()
    metrics_to_emit[DISK_METRICS_NUM_READS_PER_SEC] = read_count
    metrics_to_emit[DISK_METRICS_NUM_WRITES_PER_SEC] = write_count
    metrics_to_emit[DISK_METRICS_NUM_READ_BYTES_MIBPS] = byte_to_mib(read_bytes)
    metrics_to_emit[DISK_METRICS_NUM_WRITE_BYTES_MIBPS] = byte_to_mib(write_bytes)
    return metrics_to_emit
