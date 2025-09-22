from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Protocol
from typing import TypeVar
from typing import Union
from typing import cast

import dataclasses
import logging
import os
import wandb
from collections import OrderedDict
from interactrank.common.eval.distributed_processor import DistributedProcessor

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)
T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
Result_co = TypeVar("Result_co", covariant=True)
PartialResult = TypeVar("PartialResult")

class PlaceHolderProfiler:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def step(self) -> None:
        pass

def wandb_profile_handler_wrapper(handler: Callable, tb_log_dir: str):
    def wrapper(*args, **kwargs):
        handler(*args, **kwargs)
        if wandb.run is not None:
            wandb.save(f"{tb_log_dir}/*.pt.trace.json", base_path=tb_log_dir)
            artifact = wandb.Artifact(f"{wandb.run.id}.trace", type="profile")
            for file_path in glob.glob(f"{tb_log_dir}/*.pt.trace.json"):
                artifact.add_file(file_path)
            artifact.save()

    return wrapper
@dataclasses.dataclass
class TorchProfilerSchedule:
    """
    Configuration for the profiler schedule. The profiler will skip
    the first ``skip_first`` steps, then wait for ``wait`` steps, then do the warmup for the next ``warmup`` steps,
    then do the active recording for the next ``active`` steps and then repeat the cycle starting with ``wait`` steps.
    The optional number of cycles is specified with the ``repeat`` parameter, the zero value means that
    the cycles will continue until the profiling is finished.
    """

    skip_first: int = 100
    wait: int = 10
    warmup: int = 10
    active: int = 3
    repeat: int = 3


@dataclasses.dataclass
class TorchProfilerConfig:
    """
    record_shapes (bool): save information about operator's input shapes.
    profile_memory (bool): track tensor memory allocation/deallocation.
    with_stack (bool): record source information (file and line number) for the ops.
    with_flops (bool): use formula to estimate the FLOPs (floating point operations) of specific operators
        (matrix multiplication and 2D convolution).
    """

    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = True
    schedule: TorchProfilerSchedule = dataclasses.field(default_factory=lambda: TorchProfilerSchedule())
    with_modules: bool = False

class Accumulator(Protocol[T_contra, PartialResult, Result_co]):
    """
    Accumulates reduction result from a stream of items
    """

    def initialize(self, **kwargs: Any) -> None:
        """
        Initialize accumulator state if required. It is optional to implement this method.
        :param kwargs: Args for the initializer
        """
        pass

    def accept(self, inference_result: T_contra) -> None:
        """
        Accept an item and update the accumulator state
        :param item: Item to process
        """
        raise NotImplementedError

    def value(self) -> Result_co:
        """
        :return: Result_co of the accumulation so far
        """
        raise NotImplementedError

    def partial_value(self) -> PartialResult:
        """
        :return: Returns a representation of the accumulator state so that partial results from multiple accumulators
        can be combined to generate a final result
        """
        raise NotImplementedError

    def combine(self, partial_results: List[PartialResult]) -> Result_co:
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


InferenceResultAccumulator = Accumulator[Dict[str, Any], PartialResult, Result_co]

class Joiner(Protocol):
    def notify(self) -> None: ...

    def num_unjoined_process(self) -> int: ...

    def join(self, inference_fn: Callable[[T], Any], batch: T) -> None: ...


class SimpleJoiner(Joiner):
    def __init__(self, device, group=None) -> None:
        self.device = device
        self.group = group if group else dist.GroupMember.WORLD

    def notify(self) -> None:
        ones = torch.ones(1, device=self.device)
        dist.all_reduce(ones, group=self.group, async_op=True)

    def num_unjoined_process(self) -> int:
        logger.info("Checking number of unjoined processes, rank %d", dist.get_rank())
        num_nonjoined_procs = torch.zeros(1, device=self.device)
        dist.all_reduce(num_nonjoined_procs, group=self.group)
        return num_nonjoined_procs.item()

    def join(self, inference_fn: Callable[[T], Any], batch: T) -> None:
        while self.num_unjoined_process() != 0:
            if batch is None:
                raise RuntimeError(
                    f"There are unjoined processes but no batch to process on rank {dist.get_rank()}. "
                    f"Check your dataloader"
                )
            inference_fn(batch)


class NoOpJoiner(Joiner):
    def notify(self) -> None:
        pass

    def num_unjoined_process(self) -> int:
        return 0

    def join(self, inference_fn: Callable[[T], Any], batch: T) -> None:
        pass


class BasicEvaluator(DistributedProcessor[Dict[str, Any], Dict[str, Any]]):
    """
    Evaluator that runs the inference in a distributed fashion on multiple workers
    The evaluator takes in a list of inference result accumulators that can be used persist the predictions,
    calculate streaming metrics, etc.
    The evaluator also takes an optional post_inference_fn which is run at the end of inference.
    It can be used to calculate metrics in cases where streaming metrics cannot be computed.
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader: Iterable[T],
        inference_fn: Callable[[T, nn.Module], Dict[str, Any]],
        accumulators: Union[
            List[InferenceResultAccumulator[Any, Any]], Dict[str, InferenceResultAccumulator[Any, Any]]
        ],
        post_inference_fn: Callable[[Union[List[Any], Dict[str, Any]]], None],
        log_frequency: int = 50,
        max_num_eval_iters: Optional[int] = None,
        joiner: Optional[Joiner] = None,
        profiler_config: Optional[TorchProfilerConfig] = None,
        run_dir: Optional[str] = None,
        tb_log_dir: Optional[str] = None,
    ) -> None:
        """
        :param model: Models to run inference for
        :param dataloader: Iterable of data batches
        :param inference_fn: Callable that returns the inference result for a batch
        :param accumulators: List of accumulators to process the inference result
        :param post_inference_fn: Callable that consumes the result of the different result accumulators
        :param log_frequency: Frequency of batches at which progress is logged
        :param max_num_eval_iters: if specified, end eval after this many iterations per node (i.e. per-GPU in
            distributed GPU eval). Otherwise, if this is None or -1, continue to eval until dataloader is exhausted.
        """
        self.model = model
        self.dataloader = dataloader
        self.inference_fn = inference_fn
        self.output_dict = True
        if isinstance(accumulators, List):
            self.output_dict = False
            accumulators = {str(i): v for i, v in enumerate(accumulators)}
        self.accumulators = accumulators
        self.rank = -1
        self.post_inference_fn = post_inference_fn
        self.log_frequency = log_frequency
        self.init_args = None
        self.has_run = False
        self.max_num_eval_iters = max_num_eval_iters
        self.joiner = joiner or NoOpJoiner()
        if self.max_num_eval_iters is not None and self.max_num_eval_iters <= 0:
            logger.info(
                f"max_num_eval_iters={self.max_num_eval_iters}, this yields the same behavior as "
                "max_num_eval_iters=None, namely, the eval DataLoader will run until exhausted"
            )
            self.max_num_eval_iters = None
        self.profiler_config = profiler_config
        self.run_dir = run_dir or ""
        self.tb_log_dir = tb_log_dir or ""

    def initialize(self, rank: int, **kwargs) -> None:
        """
        :param rank: Rank of the worker running the instance of evaluator
        :param kwargs: Additional args for accumulator initializers and post_inference_fn
        """
        self.rank = rank
        for accumulator in self.accumulators.values():
            accumulator.initialize(worker_id=rank, **kwargs)

    @property
    def is_initialized(self) -> bool:
        """
        :return: True if the evaluator is properly initialized
        """
        return self.rank != -1

    def log_num_batches(self, num_batches: int):
        if num_batches % self.log_frequency == 0:
            logger.info(f"Progressed {num_batches} batches, rank {self.rank}")
            metrics = OrderedDict()
            metrics[f"Num batches {self.rank}"] = num_batches

    def run(self) -> Dict[str, Any]:
        """
        Runs inference for the model and accumulates the inference results in the given accumulators

        :return: List of partial results of all the accumulators
        """
        logger.info(f"Rank {self.rank} started running inference")
        assert not self.has_run, "Evaluator.run has already been called"
        assert self.is_initialized, "The evaluator needs to be initialized"
        self.has_run = True
        original_mode = self.model.training
        self.model.eval()

        if self.profiler_config is not None:
            profiler_config = dataclasses.asdict(self.profiler_config)
            schedule = profiler_config.pop("schedule")
            tb_log_dir = self.tb_log_dir if self.tb_log_dir else os.path.join(self.run_dir, "summary")
            rank = dist.get_rank() if dist.is_initialized() else 0
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(**schedule),
                on_trace_ready=wandb_profile_handler_wrapper(
                    torch.profiler.tensorboard_trace_handler(tb_log_dir, worker_name=f"rank_{rank}"), tb_log_dir
                ),
                **profiler_config,
            )
        else:
            prof = PlaceHolderProfiler()
        prof.start()

        stop_early = False
        with torch.no_grad():
            batch = None
            for batch_idx, batch in enumerate(self.dataloader, start=1):
                self.joiner.notify()
                inference_result = self.inference_fn(batch, self.model)
                for accumulator in self.accumulators.values():
                    accumulator.accept(inference_result)
                self.log_num_batches(batch_idx)
                # End eval early if specified, even if DataLoader is not fully exhausted.
                if self.max_num_eval_iters is not None and batch_idx >= self.max_num_eval_iters:
                    stop_early = True
                    logger.info(f"Stopping eval after completing {batch_idx} batches, rank {self.rank}")
                    break
                prof.step()
            self.joiner.join(cast(Callable[[Any], Any], lambda b: self.inference_fn(b, self.model)), batch)

        prof.stop()

        if dist.is_initialized():
            dist.barrier()

        if stop_early and hasattr(self.dataloader, "shutdown"):
            # If inference stop early, also shutdown the ongoing ray data executions to prevent resources being occupied.
            self.dataloader.shutdown()

        self.model.train(mode=original_mode)
        logger.info(f"Rank {self.rank} finished running inference")
        return {k: accumulator.partial_value() for k, accumulator in self.accumulators.items()}

    def close(self) -> None:
        for key, accumulator in self.accumulators.items():
            logger.info(f"Closing accumulator {key} on {self.rank}")
            accumulator.close()

    def merge_partial_results(self, partial_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        :param partial_results: Partial results of the different accumulators from different workers
        :return: Combined result of the different accumulators
        """
        results = {}
        for key, accumulator in self.accumulators.items():
            results[key] = accumulator.combine([partial_result[key] for partial_result in partial_results])
        return results

    def consume_result(self, result: Dict[str, Any]) -> None:
        """
        :param result: Merged result of the different accumulators
        """
        if not self.output_dict:
            result_list = [result[str(i)] for i in range(len(self.accumulators))]
            self.post_inference_fn(result_list)
        else:
            self.post_inference_fn(result)
