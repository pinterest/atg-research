from __future__ import annotations

import unittest
from unittest import mock
from unittest.mock import call

from omnisearchsage.common.logging import visualization
from omnisearchsage.common.logging.visualization import calc_global_iteration
from omnisearchsage.common.logging.visualization import write_summary
from torch import nn


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def write_summary(self, summary_writer, epoch, iteration, total_iter):
        summary_writer.add_scalar("called", "inside", "write_summary")


class VisualizationTest(unittest.TestCase):
    def test_calc_global_iter(self):
        self.assertEqual(19, calc_global_iteration(3, 4, 5))

    @mock.patch("torch.cuda.is_available")
    @mock.patch("torch.cuda.memory_allocated")
    @mock.patch("torch.cuda.max_memory_allocated")
    @mock.patch("torch.cuda.memory_reserved")
    @mock.patch("torch.cuda.max_memory_reserved")
    @mock.patch("torch.cuda.current_device")
    def test_summary(
        self,
        current_device,
        max_memory_reserved,
        memory_reserved,
        max_memory_allocated,
        memory_allocated,
        is_available,
    ):
        # set gpu metrics expected
        max_memory_reserved.return_value = 100 * 1024 * 1024 * 1024
        memory_reserved.return_value = 101 * 1024 * 1024 * 1024
        max_memory_allocated.return_value = 102 * 1024 * 1024 * 1024
        memory_allocated.return_value = 103 * 1024 * 1024 * 1024

        system_metrics = mock.MagicMock()
        system_metrics.get_gpu_metrics = mock.MagicMock(return_value=["utilization", "memory"])
        system_metrics.get_cpu_metrics = mock.MagicMock(return_value=[123, 4123, 72])
        system_metrics.get_global_network_metrics = mock.MagicMock(
            return_value=[1 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 3, 4]
        )
        system_metrics.get_global_disk_metrics = mock.MagicMock(
            return_value=[10, 11, 12 * 1024 * 1024, 13 * 1024 * 1024]
        )

        summary_writer = mock.MagicMock()

        model = DummyModel()

        write_summary(
            summary_writer,
            model,
            "batch",
            "data",
            "forward",
            "backward",
            "total_loss",
            ["loss1", "loss2"],
            0.1,
            3,
            4,
            5,
            examples_per_sec=1024,
            system_metrics=system_metrics,
            grad_norm="grad_norm",
        )

        expected = [
            call.add_scalar(visualization.GPU_METRICS_UTILIZATION, "utilization", 19),
            call.add_scalar(visualization.GPU_METRICS_MEMORY_GIB, "memory", 19),
            call.add_scalar(visualization.GPU_METRICS_MEMORY_ALLOCATED_GIB, 103, 19),
            call.add_scalar(visualization.GPU_METRICS_MEMORY_MAX_ALLOCATED_GIB, 102, 19),
            call.add_scalar(visualization.GPU_METRICS_MEMORY_RESERVED_GIB, 101, 19),
            call.add_scalar(visualization.GPU_METRICS_MEMORY_MAX_RESERVED_GIB, 100, 19),
            call.add_scalar(visualization.CPU_METRICS_PERCENT_AVAILABLE_MEM, 28.0, 19),
            call.add_scalar(visualization.CPU_METRICS_AVAILABLE_MEM_GIB, 123, 19),
            call.add_scalar(visualization.CPU_METRICS_TOTAL_MEM_GIB, 4123, 19),
            call.add_scalar(visualization.NETWORK_METRICS_BYTES_SENT_GIBPS, 8, 19),
            call.add_scalar(visualization.NETWORK_METRICS_BYTES_RECV_GIBPS, 16, 19),
            call.add_scalar(visualization.NETWORK_METRICS_BYTES_SENT_PER_SEC, 3, 19),
            call.add_scalar(visualization.NETWORK_METRICS_BYTES_RECV_PER_SEC, 4, 19),
            call.add_scalar(visualization.DISK_METRICS_NUM_READS_PER_SEC, 10, 19),
            call.add_scalar(visualization.DISK_METRICS_NUM_WRITES_PER_SEC, 11, 19),
            call.add_scalar(visualization.DISK_METRICS_NUM_READ_BYTES_MIBPS, 12, 19),
            call.add_scalar(visualization.DISK_METRICS_NUM_WRITE_BYTES_MIBPS, 13, 19),
            call.add_scalar(visualization.TRAIN_METRICS_EXAMPLES_PER_SEC, 1024, 19),
            call.add_scalar(visualization.TRAIN_METRICS_BATCH_SEC, "batch", 19),
            call.add_scalar(visualization.TRAIN_METRICS_DATA_SEC, "data", 19),
            call.add_scalar(visualization.TRAIN_METRICS_FORWARD_SEC, "forward", 19),
            call.add_scalar(visualization.TRAIN_METRICS_BACKWARD_SEC, "backward", 19),
            call.add_scalar("phase_1/learning_rate", 0.1, 19),
            call.add_scalar("phase_1/all_loss", "total_loss", 19),
            call.add_scalar("phase_1/grad_norm", "grad_norm", 19),
            call.add_scalar("iteration", 19, 19),
            call.add_scalar("phase_1/loss_0", "loss1", 19),
            call.add_scalar("phase_1/loss_1", "loss2", 19),
            call.add_scalar("called", "inside", "write_summary"),
        ]

        self.assertListEqual(summary_writer.mock_calls, expected)


if __name__ == "__main__":
    unittest.main()
