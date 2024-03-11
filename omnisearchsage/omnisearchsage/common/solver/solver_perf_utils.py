from __future__ import annotations

import enum
import time


class TimeStat:
    """
    Utility class for recording the duration of some arbitrary logic.
    """

    def __init__(self):
        self.start_time = None
        self.time_delta = None

    def start(self) -> None:
        self.start_time = time.time()

    def end(self) -> None:
        assert self.start_time is not None, "start must be called before end"
        self.time_delta = time.time() - self.start_time
        self.start_time = None

    def get(self) -> float:
        assert self.time_delta is not None, "end must be called before get"
        return self.time_delta


class ThroughputStat:
    """
    Utility class for recording the throughput of some arbitrary logic.
    """

    def __init__(self):
        self.start_time = None
        self.time_delta = None
        self.start_count = None
        self.count_delta = None

    def start(self, value: int) -> None:
        self.start_time = time.time()
        self.start_count = value

    def end(self, value: int) -> None:
        assert self.start_time is not None and self.start_count is not None, "start must be called before end"
        self.time_delta = time.time() - self.start_time
        self.count_delta = value - self.start_count
        self.start_time = self.start_count = None

    def get(self) -> float:
        assert self.time_delta is not None and self.count_delta is not None, "end must be called before get"
        return self.count_delta / self.time_delta


class TimeStatKey(enum.IntEnum):
    DATA_LOAD = 1
    FORWARD = 2
    BACKWARD = 3


class PerfTracker:
    """
    Utility class for tracking the performance metrics of mlenv solvers.

    It tracks the time spent per batch in data loading, the forward and backward passes.
    It also tracks the solver throughput over several batches.
    """

    def __init__(self):
        self.time_stats = {key: TimeStat() for key in TimeStatKey}
        self.throughput = ThroughputStat()

    def time_start(self, key: TimeStatKey) -> None:
        self.time_stats[key].start()

    def time_end(self, key: TimeStatKey) -> None:
        self.time_stats[key].end()

    def get_duration(self, key: TimeStatKey) -> float:
        return self.time_stats[key].get()

    def log_duration(self, key: TimeStatKey) -> None:
        pass

    def throughput_start(self, value: int) -> None:
        self.throughput.start(value)

    def throughput_end(self, value: int) -> None:
        self.throughput.end(value)

    def get_throughput(self) -> float:
        return self.throughput.get()

    def clock_throughput(self, value: int) -> float:
        # Equivalent to calling end, getting and restarting
        self.throughput.end(value)
        throughput = self.throughput.get()
        self.throughput.start(value)
        return throughput

    def log_throughput(self, value: float) -> None:
        pass

    def get_total_time(self):
        return sum(self.get_duration(key) for key in TimeStatKey)
