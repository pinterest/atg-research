import time

import psutil
import pynvml as nvidia_smi
from pynvml import *  # noqa: F403

MIN_GPU_MEMORY_GB = 1024 * 1024 * 1024 * 11  # 11GB


def byte_to_gb(b):
    "Byte to GB"
    return b / 1e9


def byte_to_gib(b):
    "Byte (Int) to GiB"
    return b / (1024 * 1024 * 1024)


def byte_to_mb(b):
    "Byte to GB"
    return b / 1e6


def byte_to_mib(b):
    "Byte (Int) to MiB"
    return b / (1024 * 1024)


def get_available_gpus(required_gpu_memory=MIN_GPU_MEMORY_GB):
    nvmlInit()  # noqa: F405
    deviceCount = nvmlDeviceGetCount()  # noqa: F405
    res = []
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)  # noqa: F405
        info = nvmlDeviceGetMemoryInfo(handle)  # noqa: F405
        if info.free >= required_gpu_memory:
            res.append(i)
    return res


class SystemMetric(object):
    def __init__(self):
        self.net_start = None
        self.netb_sent = None
        self.netb_recv = None
        self.netp_send = None
        self.netp_recv = None

        self.disk_start = None
        self.disk_r = None
        self.disk_w = None
        self.disk_rbytes = None
        self.disk_wbytes = None

    def _network_stats(self):
        io_results = psutil.net_io_counters()
        current_time = time.time()
        return (
            current_time,
            io_results.bytes_sent,
            io_results.bytes_recv,
            io_results.packets_sent,
            io_results.packets_recv,
        )

    def _disk_stats(self):
        disk_results = psutil.disk_io_counters()
        current_time = time.time()
        return (
            current_time,
            disk_results.read_count,
            disk_results.write_count,
            disk_results.read_bytes,
            disk_results.write_bytes,
        )

    def get_gpu_metrics(self, gpu_device):
        """
        Args:
            gpu_device: GPU device

        Returns:
            GPU utilization
            memory used in GiB
        """
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpu_device)

        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        utilization = res.gpu

        res = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        memused = byte_to_gib(res.used)

        return utilization, memused

    def get_cpu_metrics(self):
        """
        Returns:
            available physical memory in GiB
            total physical memory in GiB
            percentage of memory used
        """

        memory = psutil.virtual_memory()

        total_memory_gb = byte_to_gib(memory.total)
        available_gb = byte_to_gib(memory.available)
        percentage_used = memory.percent

        return available_gb, total_memory_gb, percentage_used

    def get_global_network_metrics(self):
        """
        Returns:
            bytes sent / sec
            bytes received / sec
            # of packages send / sec
            # of packages received / sec
        """
        if self.net_start is None:
            self.net_start, self.netb_sent, self.netb_recv, self.netp_send, self.netp_recv = self._network_stats()
            return 0.0, 0.0, 0.0, 0.0

        # current timings
        current_time, netb_sent, netb_recv, netp_send, netp_recv = self._network_stats()
        elapse_time = current_time - self.net_start

        # compare with last measure
        netb_srate = (netb_sent - self.netb_sent) / elapse_time
        netb_rrate = (netb_recv - self.netb_recv) / elapse_time
        netp_srate = (netp_send - self.netp_send) / elapse_time
        netp_rrate = (netp_recv - self.netp_recv) / elapse_time

        # update timing
        self.net_start, self.netb_sent, self.netb_recv, self.netp_send, self.netp_recv = (
            current_time,
            netb_sent,
            netb_recv,
            netp_send,
            netp_recv,
        )
        return netb_srate, netb_rrate, netp_srate, netp_rrate

    def get_global_disk_metrics(self):
        """
        Returns:
            read bytes / sec
            write bytes / sec
            number of reads / sec
            number of writes / sec
        """
        if self.disk_start is None:
            self.disk_start, self.disk_r, self.disk_w, self.disk_rbytes, self.disk_wbytes = self._disk_stats()
            return 0.0, 0.0, 0.0, 0.0

        # current timings
        current_time, disk_r, disk_w, disk_rbytes, disk_wbytes = self._disk_stats()
        elapse_time = current_time - self.disk_start

        # compare with last measure
        disk_rrate = (disk_r - self.disk_r) / elapse_time
        disk_wrate = (disk_w - self.disk_w) / elapse_time
        disk_rbytes_rate = (disk_rbytes - self.disk_rbytes) / elapse_time
        disk_wbytes_rate = (disk_wbytes - self.disk_wbytes) / elapse_time

        # update timing
        self.disk_start, self.disk_r, self.disk_w, self.disk_rbytes, self.disk_wbytes = (
            current_time,
            disk_r,
            disk_w,
            disk_rbytes,
            disk_wbytes,
        )
        return disk_rrate, disk_wrate, disk_rbytes_rate, disk_wbytes_rate
