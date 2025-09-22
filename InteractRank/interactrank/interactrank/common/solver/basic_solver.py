from __future__ import annotations

from typing import IO
from typing import TYPE_CHECKING
from typing import Any
from typing import BinaryIO
from typing import Callable
from typing import Deque
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import cast
from typing_extensions import Protocol

import contextlib
import copy
import dataclasses
import inspect
import io
import json
import logging
import os
import sys
import tempfile
import time
from abc import ABC
from collections import defaultdict
from collections import deque
from functools import cached_property
from functools import partial
from statistics import mean
from math import isnan

import smart_open
import torch
import torch.distributed as dist
import torchinfo
import torchsnapshot
from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from interactrank.common.solver.solver import is_root_process
from interactrank.loggingcontent.visualization import write_summary
from interactrank.common.solver.solver_perf_utils import PerfTracker
from interactrank.common.solver.solver_perf_utils import TimeStatKey
from interactrank.common.solver.solver import Solver
from interactrank.common.eval.basic_evaluator import BasicEvaluator
from interactrank.common.eval.distributed_processor import TorchDistributedRunner

from interactrank.common.utils.utils import mkdir_p

if TYPE_CHECKING:
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data import DataLoader

T = TypeVar("T")
V = TypeVar("V")
_ModelReturnType = Dict[str, Union[Tensor, Dict[str, Tensor], List[Tensor]]]
_ModelForwardFuncT = Callable[[nn.Module, T], _ModelReturnType]

RUN_EVAL_MLFLOW_TAG = "run_eval"
LATEST_SNAPSHOT_FILENAME = "snapshot_latest.pth"
LOCAL_SNAPSHOT_PATH = f"/tmp/{LATEST_SNAPSHOT_FILENAME}"
S3_SNAPSHOT_SUB_DIR = "snapshots"


logger = logging.getLogger(__name__)

class PlaceHolderProfiler:
    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def step(self) -> None:
        pass
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
    active: int = 10
    repeat: int = 0

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
    schedule: TorchProfilerSchedule = TorchProfilerSchedule()
    with_modules: bool = False


class EvalFunc(Protocol):
    def __call__(
        self,
        run_dir: str,
        iteration: int,
        summary_writer: Optional[SummaryWriter],
        eval_every_n_iter: int,
        tb_log_dir: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]: ...

    @staticmethod
    def from_evaluator_dict(
            evaluator_dict: Dict[str, Callable[..., BasicEvaluator]], results_in_all_procs: bool = False
    ) -> EvalFunc:
        def fn(
                run_dir: str,
                iteration: int,
                summary_writer: Optional[SummaryWriter],
                eval_every_n_iter: int,
                tb_log_dir: Optional[str] = None,
        ) -> Optional[Dict[str, Any]]:
            args = dict(
                run_dir=run_dir,
                iteration=iteration,
                summary_writer=summary_writer,
                eval_every_n_iter=eval_every_n_iter,
                tb_log_dir=tb_log_dir,
            )
            result = {}
            for evaluator_key, evaluator_fn in evaluator_dict.items():
                result[evaluator_key] = TorchDistributedRunner(evaluator_fn(**args), results_in_all_procs)(**args)
            return result

        return fn

    @staticmethod
    def from_eval_fn(eval_fn: Callable, iterations: int, eval_in_all_procs: bool = False) -> EvalFunc:
        def wrapper(
            run_dir: str,
            iteration: int,
            summary_writer: Optional[SummaryWriter],
            eval_every_n_iter: int,
            tb_log_dir: Optional[str] = None,
        ) -> None:
            eval_dir = os.path.join(run_dir, "eval")
            if eval_in_all_procs:
                eval_fn(
                    run_dir=eval_dir,
                    iteration=iteration,
                    summary_writer=summary_writer,
                    run_full_eval=iteration == iterations,
                    tb_log_dir=tb_log_dir,
                )
            else:
                if dist.get_rank() == 0:
                    eval_fn(
                        run_dir=eval_dir,
                        iteration=iteration,
                        summary_writer=summary_writer,
                        run_full_eval=iteration == iterations,
                        tb_log_dir=tb_log_dir,
                    )
                dist.barrier()

        return wrapper


class MLEnvSolverBase(Solver, Generic[T, V], ABC):
    TOTAL_LOSS = "total_loss"
    LOSS_COMPONENTS = "loss_components"

    _SummarizeFuncT = Callable[[_ModelReturnType], V]
    _LogSummaryFuncT = Callable[[int, int, V, Optional[SummaryWriter]], None]

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataset_loader: DataLoader,
        lr_scheduler: _LRScheduler,
        batch_size: int,
        snapshot_every_n_iter: int = 250000,
        snapshot_filename_prefix: str = "snap",
        snapshot_dir: Optional[str] = None,
        extra_repr_info: Optional[Dict[str, Any]] = None,
        save_feature_map_func: Optional[Callable[[str, bool], None]] = None,
        steps: int = 250000,
        model_forward_func: _ModelForwardFuncT = lambda m, b: m(b),
        summarize_every_n_iter: int = 50,
        summarize_func: _SummarizeFuncT = lambda x: x,
        compute_summary_from_all_processes: bool = False,
        eval_every_n_iter: int = 0,
        eval_at_end: bool = True,
        eval_at_start: bool = False,
        eval_func: Optional[EvalFunc] = None,
        max_grad_norm: Optional[float] = None,
        precision: torch.dtype = torch.float32,
        save_lr_scheduler_states: Optional[bool] = False,
        log_summary_metrics_func: Optional[_LogSummaryFuncT] = None,
        torchscript_deploy_fn: Optional[Callable[[torch.nn.Module, str], None]] = None,
        calc_feature_monitoring_stats_func: Optional[Callable[[str, bool], None]] = None,
        post_train_step_func: Optional[Callable[[_ModelReturnType, Deque], None]] = None,
        profiler_config: Optional[TorchProfilerConfig] = None,
        gradient_accumulation_steps: int = 1,
        deploy_best_model: bool = False,
        iterations: int = sys.maxsize,
        epochs: int = sys.maxsize,
        snapshot_every_n_epochs: int = 0,
        eval_every_n_epochs: int = 0,
        early_stop_func_check_by_iter: bool = False,
        early_stop_func_check_by_epoch: bool = False,
        enable_context_join: bool = False,
        divide_by_initial_world_size: bool = True,
        post_epoch_func=None,
        start_iter: int = 0,
        enable_flops: bool = False,
        log_grad_norm: bool = False,
    ):
        """
        General Args:
            model: Model to train
            optimizer: Optimizer
            train_dataset_loader: Train dataset DataLoader
            lr_scheduler: _LRScheduler
            batch_size: Batch size used during training
            max_grad_norm: (Optional) Max norm to clip the gradient. Default: No gradient clipping is done
            precision: The precision to use during training. Available options:
                [torch.float32, torch.float16, torch.bfloat16].
                Default to torch.float32 which means mixed precision is disabled.
            save_lr_scheduler_states: Save lr scheduler states in save_snapshot if True.
            snapshot_every_n_iter: Saves snapshot every n iterations
            snapshot_filename_prefix: Filename prefix to use for snapshotting
            snapshot_dir: Directory to store the snapshots, can be local or s3 dir. Optional, by default they are stored
            in the run directory
            extra_repr_info: Extra information that should affect the solver hash (for example, eval data source)
            gradient_accumulation_steps: The number of steps that should pass before gradients are accumulated.
            early_stop_func: callable to determine whether to early stop training loop
            deploy_best_model: whether to deploy the best model after early stopping
            iterations: Num of iterations (an iteration is one gradient update based on a minibatch of examples) to run
            epochs: Num of epochs to run
            snapshot_every_n_epochs: Saves snapshot every n epochs
            start_iter: Starting iteration number, useful for resuming training from a iteration based snapshot
            enable_flops: Enable Flop per second calculation.
            log_grad_norm: Whether to log the gradient norm (when max_grad_norm is specified, this flag will be ignored and grad norm will always be logged)

        Pre-execution Hooks:
            save_feature_map_func: Customizes how the feature map should be saved

        Train Step Hooks:
            model_forward_func: Customizes how the batch is passed to the model forward pass

            eval_every_n_iter: Runs eval_func every n iterations
            eval_every_n_epochs: Run eval_func every n epochs
            eval_at_end: Flag to indicate if eval_func should be run at the end of training. Note that if f `early_stop_func` is defined and `deploy_best_model` is enabled, turning on this flag will re-evaluate using the best snapshot during the finalization steps.
            eval_at_start: Flag to indicate if eval_func should be run at the start of training
            eval_func: Customizes model evaluation
            early_stop_func_check_by_iter: Flag to check stop func in iterations
            early_stop_func_check_by_epoch: Flag to check stop func at the end of epochs
            enable_context_join: Flat to enable join context manager to handle uneven inputs on different devices
            divide_by_initial_world_size: Only applies to distributed training and when enable_context_join=True.
                If true, at the end of training when
                fewer than world_size processes still participate in training (because some have finished their
                allocated data shards), the gradients will continue to be divided by world_size. If false, the gradients
                are divided by the remaining number of active processes.
            post_epoch_update_func: function to apply at the end of each epoch

            summarize_func: Customizes logging summary_metrics (model output)
            summarize_every_n_iter: Logs summary_metrics every n iterations
            compute_summary_from_all_processes: If True, summary_metrics will be computed from all processes


        Post-execution Hooks:
            torchscript_deploy_fn: Customizes generating and saving model torch script
            calc_feature_monitoring_stats_func: Customizes calculating feature monitoring stats
            post_train_step_func: Customizes model training metrics checks
        """

        super().__init__(
            model=model,
            optimizer=optimizer,
            eval_every_n_iter=eval_every_n_iter,
            steps=steps,
            lr_scheduler=lr_scheduler,
            extra_repr_info=extra_repr_info,
        )
        # General Args
        self.model = model
        self.optimizer = optimizer
        self.train_dataset_loader = train_dataset_loader
        self.lr_scheduler = lr_scheduler
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.log_grad_norm = log_grad_norm
        self.set_zero_grad_to_none = (
            "set_to_none" in inspect.signature(self.optimizer.zero_grad).parameters if self.optimizer else False
        )

        self.snapshot_every_n_iter = snapshot_every_n_iter
        self.snapshot_filename_prefix = snapshot_filename_prefix
        self.snapshot_save_dir = snapshot_dir

        # Pre-execution Hooks
        self.save_feature_map_func = save_feature_map_func

        # Train Step Hooks
        self.model_forward_func = model_forward_func

        assert summarize_func is None or summarize_every_n_iter > 0, "summarize_every_n_iter must be defined"
        self.summarize_every_n_iter = summarize_every_n_iter
        self.summarize_func = summarize_func
        self.compute_summary_from_all_processes = compute_summary_from_all_processes

        self.eval_every_n_iter = eval_every_n_iter
        self.eval_at_end = eval_at_end
        self.eval_at_start = eval_at_start
        self.eval_func = eval_func
        self.post_epoch_func = post_epoch_func

        self.log_summary_metrics_func = log_summary_metrics_func

        # Post-execution Hooks
        self.torchscript_deploy_fn = torchscript_deploy_fn
        self.calc_feature_monitoring_stats_func = calc_feature_monitoring_stats_func
        self.post_train_step_func = post_train_step_func

        self.profiler_config = profiler_config

        # Additional class member variables
        self.summary_writer: Optional[SummaryWriter] = None
        self.start_iter = 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.stopped_early = False

        self.loss_component_history: Dict[str, Deque[Tensor]] = defaultdict(lambda: deque(maxlen=100))
        self.total_loss_history: Deque = deque(maxlen=100)
        self.grad_norm_history: Deque[Tensor] = deque(maxlen=100)

        assert precision in {
            torch.float32,
            torch.float16,
            torch.bfloat16,
        }, f"Training precision must be float32, float16, or bfloat16. {precision} is found."
        self.precision = precision
        self.enable_mixed_precision = self.precision != torch.float32
        if self.enable_mixed_precision:
            logger.info(f"Training with mixed precision using {self.precision}")
        # Bfloat16 don't need to scale gradient
        self.scaler = torch.cuda.amp.GradScaler() if self.precision == torch.float16 else None

        self.save_lr_scheduler_states = save_lr_scheduler_states

        self.perf_tracker = PerfTracker()

        self._cuda_perf_tracker = None

        self._log_model_summary()

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.deploy_best_model = deploy_best_model

        # For BasicSolver self.epochs is set to inf, and for EpochSolver self.iterations is set to inf
        self.iterations = iterations
        self.start_iter = start_iter
        self.epochs = epochs
        self.snapshot_every_n_epochs = snapshot_every_n_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.early_stop_func_check_by_iter = early_stop_func_check_by_iter
        self.early_stop_func_check_by_epoch = early_stop_func_check_by_epoch
        self.enable_context_join = enable_context_join
        self.divide_by_initial_world_size = divide_by_initial_world_size
        self.enable_flops = enable_flops

    def _log_model_summary(self) -> None:
        summary = torchinfo.summary(self.model, depth=5)
        if is_root_process():
            print(summary)

    @cached_property
    def is_root_process(self) -> bool:
        return is_root_process()

    @staticmethod
    def load_pretrain_weights(
        model: nn.Module,
        snapshot_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        strict_load: bool = True,
        checkpoint_process_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Loads model weights from the specified path
        :param model: Model to load the weights to
        :param snapshot_file_path: Local path of the snapshot.
        :param strict_load: if set False, will ignore the non-matching keys.
        :param checkpoint_process_fn: Function to process the state_dict before loading
        """
        checkpoint = torch.load(snapshot_file_path, map_location="cpu")
        if checkpoint_process_fn:
            checkpoint = checkpoint_process_fn(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=strict_load)

    @staticmethod
    def load_pretrain_weights_and_optimizer_states_and_lr_scheduler(
        model: nn.Module,
        optimizer: nn.Module,
        lr_scheduler: nn.Module,
        snapshot_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        checkpoint_process_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Loads model weights, optimizer states and lr scheduler states from the specified path
        :param model: Model to load the weights to
        :param optimizer: Optimizer to load the optimizer states to
        :param lr_scheduler: LR scheduelr to load the lr scheduler states to
        :param snapshot_file_path: Local path of the snapshot.
        :param state_dict_process_fn: Function to process the state_dict before loading
        """
        checkpoint = torch.load(snapshot_file_path, map_location="cpu")
        if checkpoint_process_fn:
            checkpoint = checkpoint_process_fn(checkpoint)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    @staticmethod
    def load_distributed_pretrain_weights(
        model: nn.Module, snapshot_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]]
    ) -> None:
        """
        Loads model weights from the specified path
        :param model: Model to load the weights to
        :param snapshot_file_path: Local path of the snapshot.
        """
        checkpoint = torch.load(snapshot_file_path, map_location="cpu")
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
        model.load_state_dict(checkpoint["state_dict"], strict=True)

    @staticmethod
    def load_partial_pretrain_weights(
        model: nn.Module,
        snapshot_file_path: Union[str, os.PathLike, BinaryIO, IO[bytes]],
        module_shift_map: Optional[Dict[str, str]] = None,
        partial_params_to_load_func: Optional[Callable[[str], bool]] = None,
        strict: bool = True,
        ensure_size_match: bool = False,
        checkpoint_process_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Loads partial pre-trained weights into the model.

        Args:
            model (nn.Module): The model to load the weights into.
            snapshot_file_path (Union[str, os.PathLike, BinaryIO, IO[bytes]]): The path to the snapshot file containing the pre-trained weights.
            module_shift_map (dict, optional): Mapping old_module_prefix to new_module_prefix. Replacing old_module_prefix in module names by new_module_prefix.
            partial_params_to_load_func(callable, optional): Whether to load the partial parameters based on the function.
            strict (bool, optional): Whether to strictly enforce that the keys in the checkpoint match the keys in the model's state_dict. Defaults to True.
            ensure_size_match (bool, optional): Whether to ensure that the sizes of the pre-trained weights match the sizes of the corresponding model weights. Defaults to False.

        Returns:
            Dict[str, Any]: A dictionary containing the loaded pre-trained weights.
        """
        checkpoint = torch.load(snapshot_file_path, map_location="cpu")
        if checkpoint_process_fn:
            checkpoint = checkpoint_process_fn(checkpoint)
        model_dict = model.state_dict()
        if module_shift_map is not None:
            new_checkpoint_state_dict = {}
            for module_name, param_val in checkpoint["state_dict"].items():
                new_module_name = module_name
                for old_module_prefix, new_module_prefix in module_shift_map.items():
                    if module_name.startswith(old_module_prefix):
                        new_module_name = module_name.replace(old_module_prefix, new_module_prefix)
                        break
                new_checkpoint_state_dict[new_module_name] = param_val
            checkpoint_state_dict = new_checkpoint_state_dict
        else:
            checkpoint_state_dict = checkpoint["state_dict"]
        if partial_params_to_load_func is not None:
            checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if partial_params_to_load_func(k)}
        pretrained_dict = {
            k: v for k, v in checkpoint_state_dict.items() if k in model_dict and v.size() == model_dict[k].size()
        }
        logger.info(f"Loading {pretrained_dict.keys()} keys from checkpoint")
        # unmatched keys: keys exisiting in pretrained model but unmatched in size with the model, will throw an error if ensize_size_match is True
        unmatched_keys = [
            k
            for k, v in model_dict.items()
            if k in checkpoint_state_dict and v.size() != checkpoint_state_dict[k].size()
        ]
        if not ensure_size_match:
            logger.info(f"{unmatched_keys} are not loaded due to size mismatch")
        else:
            if len(unmatched_keys) > 0:
                raise KeyError(
                    f"{unmatched_keys} are not loaded due to same module name but mismatched size, set ensure_size_match=False to ignore if this is expected"
                )

        # unloaded keys: keys not existing in the pretrained model or filtered out by partial_params_to_load_func
        unloaded_keys = [k for k in model_dict if k not in checkpoint_state_dict]
        logger.info(
            f"{unloaded_keys} are not loaded due not existing in the base model or filtered out by partial_params_to_load_func"
        )
        model.load_state_dict(pretrained_dict, strict=strict)

        return pretrained_dict

    def _is_gradient_accumulating(self, current_iter: int) -> bool:
        """Return true if we are still accumulating the gradients and should not run step

        Args:
            current_iter (int): current iteration number
        """
        return (current_iter + 1) % self.gradient_accumulation_steps != 0

    def load_snapshot(self, snapshot_file_path: str) -> None:
        """
        Loads the snapshot from the specified path
        :param snapshot_file_path: Local path of the snapshot.
        """
        checkpoint = torch.load(snapshot_file_path, map_location="cpu")
        self.start_iter = checkpoint["iter"]
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info(f"Loading snapshot from {snapshot_file_path}, resume training from {self.start_iter} iteration")

    def _snapshot_prefix(self, save_dir: str) -> str:
        snapshots_dir = os.path.join(save_dir, S3_SNAPSHOT_SUB_DIR)
        mkdir_p(snapshots_dir)
        return os.path.join(snapshots_dir, self.snapshot_filename_prefix)

    def save_best_eval_snapshot(self, best_iter: int, snapshot_prefix: str) -> None:
        """This function is called when we save the snapshot which has the best eval result.
        By default, we call save_snapshot directly. Have this as independent function in case some
        trainers only want to save the model with the best eval result.
        """
        return self.save_snapshot(best_iter, self.get_best_model_snapshot_path(snapshot_prefix))

    def save_snapshot(
        self,
        iteration: int,
        snapshot_filename: str,
        batch: Optional[T] = None,
        root_process_only: bool = True,
    ) -> None:

        _to_save = {
            "iter": iteration,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        if self.save_lr_scheduler_states:
            lr_scheduler_to_save = copy.deepcopy(self.lr_scheduler.state_dict())
            lr_scheduler_to_save.pop("after_scheduler_fn", None)
            _to_save["lr_scheduler"] = lr_scheduler_to_save
        if batch is not None:
            _to_save["batch"] = batch

        if is_root_process() or not root_process_only:
            # save checkpoint
            with smart_open.open(snapshot_filename, "wb") as f:
                torch.save(_to_save, f)

    def run_eval(self, run_dir: str, iteration: int) -> Optional[Dict[str, Any]]:
        """
        Run the given evaluation function. This is run from all gpu processes since the compute is distributed
        and synchronized.

        Args:
            run_dir: Run directory of the solver
            iteration: Current iteration. Used for file naming
        """
        if self.eval_func is None:
            return

        # evaluate on validation set
        logger.info(f"Evaluate on evaluation set at {iteration} iteration")
        return self.eval_func(
            run_dir=run_dir,
            iteration=iteration,
            summary_writer=self.summary_writer,
            eval_every_n_iter=self.eval_every_n_iter,
        )

    def run_train_init_steps(self, run_dir: str, overwrite: bool, tb_log_dir: Optional[str]) -> None:
        """
        Performs initialization prior to executing the main training loop.

        Args:
            run_dir: Run directory of the solver
            overwrite: whether to overwrite existing training runs in dir
            tb_log_dir: local or s3 directory to write Tensorboard records to
        """
        if self.save_feature_map_func and self.is_root_process:
            self.save_feature_map_func(run_dir, overwrite)

        if self.is_root_process:
            self.summary_writer = SummaryWriter(log_dir=tb_log_dir if tb_log_dir else os.path.join(run_dir, "summary"))

        if self.eval_at_start:
            self.run_eval(run_dir, self.start_iter)

        if dist.is_initialized():
            dist.barrier()

        # switch to train mode
        self.model.train()

        self.optimizer.zero_grad()

    def log_process(
        self,
        *,
        iteration: int,
        effective_iteration: int,
        lr: List[float],
        data_load_time: float,
        forward_time: float,
        backward_time: float,
        batch_time: float,
        examples_per_sec: float,
        total_loss: float,
        grad_norm: float,
        loss_components: Dict[str, float],
        flop_per_sec: Optional[float] = None,
    ) -> None:
        """
        Log the training progress

        Args:
            iteration:              Current Iteration
            effective_iteration:    Current Iteration * world size
            lr:                     Current Learning Rate
            data_load_time:         Data loading time
            forward_time:           Forward loading time
            backward_time:          Backward loading time
            batch_time:             Total batch time
            examples_per_sec:       Examples per second
            total_loss:             Total Loss
            grad_norm:              Gradient norm
            loss_components:        Components of total loss
            flop_per_sec:           Flop per Second
        """
        logger.info(
            f"Iteration: {iteration}\t"
            f"LR: {[f'{lr_i:.9f}' for lr_i in lr]}\t"
            f"Data: {data_load_time:.6f}\t"
            f"Forward: {forward_time:.6f}\t"
            f"Backward: {backward_time:.6f}\t"
            f"Batch Time: {batch_time:.6f}\n"
            f"Examples per Second: {examples_per_sec:.6f}\n"
            f"Grad Norm: {grad_norm:.6f}\n"
            f"Total Loss: {total_loss:.6f}\n"
            f"TFLOPS: {flop_per_sec / 1e12:.6f}\n"
            if flop_per_sec
            else ""
        )
        if loss_components:
            print(f"Loss Components: {[f'{name}={loss:.6f}' for name, loss in loss_components.items()]}")

        if self.summary_writer:
            write_summary(
                summary_writer=self.summary_writer,
                model=self.model,
                batch_time=batch_time,
                data_time=data_load_time,
                forward_time=forward_time,
                backward_time=backward_time,
                total_loss=total_loss,
                per_task_loss=loss_components,
                learning_rate=lr,
                epoch=0,
                iteration=iteration,
                examples_per_sec=examples_per_sec,
                grad_norm=grad_norm,
            )

    def log_summary(self, current_iter: int, summary_metrics: V, flop_per_sec: Optional[float] = None) -> None:
        """
        Logs a summary of the training process at a given iteration.

        Args:
            current_iter (int): The current iteration number.
            summary_metrics (V): Summary metrics to log.
            flop_per_sec (Optional[float]): Floating point operations per second.

        Returns:
            None
        """
        world_size = 1 if not dist.is_initialized() else dist.get_world_size()
        effective_iteration: int = current_iter * world_size
        examples_per_sec = self.perf_tracker.clock_throughput(effective_iteration * self.batch_size)

        self.log_process(
            iteration=current_iter,
            effective_iteration=effective_iteration,
            lr=self.lr_scheduler.get_last_lr(),
            data_load_time=self.perf_tracker.get_duration(key=TimeStatKey.DATA_LOAD),
            forward_time=self.perf_tracker.get_duration(key=TimeStatKey.FORWARD),
            backward_time=self.perf_tracker.get_duration(key=TimeStatKey.BACKWARD),
            batch_time=self.perf_tracker.get_total_time(),
            examples_per_sec=examples_per_sec,
            total_loss=mean(self.total_loss_history or [0]),
            loss_components={
                name: cast(float, torch.stack(list(loss_history)).mean(dtype=torch.float32).item())
                for name, loss_history in self.loss_component_history.items()
            },
            grad_norm=cast(
                float, torch.stack(list(self.grad_norm_history) or [torch.tensor(0)]).mean(dtype=torch.float32).item()
            )
        )
        self.perf_tracker.log_throughput(value=examples_per_sec)

        if self.log_summary_metrics_func is not None:
            self.log_summary_metrics_func(current_iter, effective_iteration, summary_metrics, self.summary_writer)

    def nan_loss_debug(self, current_iter: int, batch: T) -> None:
        """
        Debugging function to help identify the source of NaNs in the loss.

        :param batch: A batch of examples
        """
        activation = {}

        def _extract_metadata(output: torch.Tensor):
            return {
                "shape": output.shape,
                "max": output.max(),
                "min": output.min(),
                "mean": output.mean(),
                "std": output.std(),
            }

        def hook(model: nn.Module, input: T, output: T, name: str):
            if isinstance(output, torch.Tensor) and output.is_floating_point():
                activation[name] = _extract_metadata(output)

            elif isinstance(output, dict):
                activation[name] = {}
                for k, v in output.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        activation[name][str(k)] = _extract_metadata(v)

            elif isinstance(output, (list, tuple)):
                activation[name] = []
                for v in output:
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        activation[name].append(_extract_metadata(v))

        for name, module in self.model.named_modules():
            if not isinstance(module, torch.jit.ScriptModule):
                module.register_forward_hook(partial(hook, name=name))

        with torch.no_grad():
            self.model_forward_func(self.model, batch)

        # Log activation metadata, snapshot and batch
        with tempfile.TemporaryDirectory() as tmpdir:
            activation_path = os.path.join(tmpdir, "activation_log.json")
            with open(activation_path, "w") as tmp:
                json.dump(activation, tmp, indent=4)

            snapshot_with_batch_path = os.path.join(tmpdir, self.snapshot_filename_prefix + "_final.pth")
            self.save_snapshot(
                iteration=current_iter,
                snapshot_filename=snapshot_with_batch_path,
                batch=batch,
                root_process_only=False,
            )

    def _measure_flops(self, current_iter: int) -> bool:
        """
        Determines whether FLOPs measurement should be performed.

        Args:
            current_iter (int): The current iteration number in the loop.

        Returns:
            bool: True if FLOPs should be measured at the current iteration,
                False otherwise.
        """
        return self.enable_flops and self.is_root_process and current_iter % self.summarize_every_n_iter == 0

    def _backward_pass_and_optimize(self, total_loss: Tensor, current_iter: int) -> None:
        """
        Perform the backward pass and optimization step for the model.

        Args:
            total_loss (Tensor): The total loss computed for the current batch.
            current_iter (int): The current iteration number.

        Returns:
            None
        """
        skip_lr_sched = False

        # We will only do optimizer and lr scheduler step if we are not accumulating the gradients
        not_gradient_accumulating = not self._is_gradient_accumulating(current_iter)

        if self.scaler is not None:
            # Used with fp16, for context see https://pytorch.org/docs/stable/amp.html#gradient-scaling
            scale = self.scaler.get_scale()
            self.scaler.scale(total_loss / self.gradient_accumulation_steps).backward()

            if not_gradient_accumulating:
                # Unscales the gradients of optimizer's assigned params in-place
                self.scaler.unscale_(self.optimizer)
                if self.max_grad_norm is not None or self.log_grad_norm:
                    max_grad_norm = self.max_grad_norm or sys.float_info.max
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm, norm_type=2.0
                    )
                    self.grad_norm_history.append(grad_norm.mean().detach())
                # Skips optimizer.step() if the gradients contain infs or NaNs.
                self.scaler.step(self.optimizer)
                self.scaler.update()
                # The scale factor often causes infs/NaNs to appear in gradients for the first few iterations
                # as its value calibrates. scaler.step will skip the underlying optimizer.step() for these iterations.
                # After that, step skipping should occur rarely (once every few hundred or thousand iterations).
                # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/2
                skip_lr_sched = scale > self.scaler.get_scale()
        else:
            (total_loss / self.gradient_accumulation_steps).backward()

            if not_gradient_accumulating:
                if self.max_grad_norm is not None or self.log_grad_norm:
                    max_grad_norm = self.max_grad_norm or sys.float_info.max
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm, norm_type=2.0
                    )
                    self.grad_norm_history.append(grad_norm.mean().detach())
                self.optimizer.step()

        if not_gradient_accumulating:
            if self.set_zero_grad_to_none:
                self.optimizer.zero_grad(set_to_none=True)
            else:
                self.optimizer.zero_grad()

        if not_gradient_accumulating and self.lr_scheduler and not skip_lr_sched:
            self.lr_scheduler.step()

    def _append_loss_history(self, total_loss_value: float, metrics: Dict[str, Tensor]) -> None:
        """
        Appends the total loss value and loss components to their respective
        history lists.

        Args:
            total_loss_value (float): The total loss value to be appended.
            metrics (Dict[str, Tensor]): A dictionary containing various metrics,
                including loss components.

        Returns:
            None
        """
        self.total_loss_history.append(total_loss_value)

        if self.LOSS_COMPONENTS in metrics:
            loss_components = cast(Union[List[Tensor], Dict[str, Tensor]], metrics[self.LOSS_COMPONENTS])
            if isinstance(loss_components, list):
                loss_components = {f"loss_{i}": loss for i, loss in enumerate(loss_components)}
            for name, loss in loss_components.items():
                self.loss_component_history[name].append(loss.detach())

    def _assert_no_nan_in_loss(self, total_loss: Tensor, current_iter: int, batch: T) -> None:
        """
        Asserts that the total loss does not contain NaN values.

        Args:
            total_loss (Tensor): The total loss tensor.
            current_iter (int): The current iteration number.
            batch (T): The current batch of data.

        Raises:
            ValueError: If the total loss contains NaN values.
        """
        if torch.isnan(total_loss):
            total_loss.detach_()
            try:
                self.nan_loss_debug(current_iter, batch)
            finally:
                raise ValueError("Training loss cannot be Nan.")

    def _forward_pass(self, current_iter: int, batch: T) -> Tuple[Dict[str, Tensor], Optional[V]]:
        """
        Perform a forward pass through the model and optionally compute
        summary metrics.

        Args:
            current_iter (int): The current iteration number.
            batch (T): The input batch of data.

        Returns:
            Tuple[Dict[str, Tensor], Optional[V]]: A tuple containing:
                - metrics (Dict[str, Tensor]): The computed metrics from the
                      forward pass.
                - summary_metrics (Optional[V]): The summary metrics if computed,
                      otherwise None.
        """
        with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision, dtype=self.precision):
            # Conduct the forward computation
            metrics = self.model_forward_func(self.model, batch)

            # Conditionally compute summary metrics
            summary_metrics = None
            if (
                self.is_root_process or self.compute_summary_from_all_processes
            ) and current_iter % self.summarize_every_n_iter == 0:
                with torch.no_grad():
                    summary_metrics = self.summarize_func(metrics)

        return metrics, summary_metrics

    def run_train_step(self, current_iter: int, batch: T) -> None:
        """
        Execute one step in the training loop: forward pass, backward pass, optimizer update and optionally log
        additional summary and throughput related metrics.

        :param current_iter: Current iteration
        :param batch: A batch of examples
        """
        self.perf_tracker.time_start(key=TimeStatKey.FORWARD)
        skip_lr_sched = False
        with torch.cuda.amp.autocast(enabled=self.enable_mixed_precision, dtype=self.precision):
            metrics = self.model_forward_func(self.model, batch)
            if (
                is_root_process() or self.compute_summary_from_all_processes
            ) and current_iter % self.summarize_every_n_iter == 0:
                with torch.no_grad():
                    summary_metrics = self.summarize_func(metrics)
            else:
                summary_metrics = None

        total_loss = cast(Tensor, metrics[self.TOTAL_LOSS])
        total_loss_value = total_loss.item()
        if isnan(total_loss_value):
            total_loss.detach_()
            try:
                self.nan_loss_debug(current_iter, batch)
            finally:
                raise ValueError("Training loss cannot be Nan.")
        self.total_loss_history.append(total_loss_value)
        if self.LOSS_COMPONENTS in metrics:
            loss_components = cast(Union[List[Tensor], Dict[str, Tensor]], metrics[self.LOSS_COMPONENTS])
            if isinstance(loss_components, list):
                loss_components = {f"loss_{i}": loss for i, loss in enumerate(loss_components)}
            for name, loss in loss_components.items():
                self.loss_component_history[name].append(loss.detach())

        self.perf_tracker.time_end(key=TimeStatKey.FORWARD)
        self.perf_tracker.time_start(key=TimeStatKey.BACKWARD)
        if self.scaler is not None:
            # Used with fp16, for context see https://pytorch.org/docs/stable/amp.html#gradient-scaling
            scale = self.scaler.get_scale()
            self.scaler.scale(total_loss).backward()

            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer)

            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type=2.0)
                self.grad_norm_history.append(grad_norm.mean())

            # Skips optimizer.step() if the gradients contain infs or NaNs.
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # The scale factor often causes infs/NaNs to appear in gradients for the first few iterations
            # as its value calibrates. scaler.step will skip the underlying optimizer.step() for these iterations.
            # After that, step skipping should occur rarely (once every few hundred or thousand iterations).
            # https://discuss.pytorch.org/t/optimizer-step-before-lr-scheduler-step-error-using-gradscaler/92930/2
            skip_lr_sched = scale > self.scaler.get_scale()
        else:
            total_loss.backward()

            if self.max_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm, norm_type=2.0)
                self.grad_norm_history.append(grad_norm.mean().detach())
            self.optimizer.step()

        if self.set_zero_grad_to_none:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            self.optimizer.zero_grad()

        self.perf_tracker.time_end(key=TimeStatKey.BACKWARD)

        # https://discuss.pytorch.org/t/calling-loss-backward-reduce-memory-usage/2735/2
        del total_loss

        if self.lr_scheduler and not skip_lr_sched:
            self.lr_scheduler.step()

        # Log duration metrics to statsboard for every rank
        for key in TimeStatKey:
            self.perf_tracker.log_duration(key)

        if summary_metrics is not None and is_root_process():
            self.log_summary(current_iter, summary_metrics)

        if self.post_train_step_func:
            self.post_train_step_func(metrics, self.total_loss_history)

    def run_train_finalize_steps(self, iteration: int, run_dir: str, overwrite: bool) -> None:
        """
        Performs the final steps after the training loop is completed (saving model weights, final evaluation,
        torchscript conversion, feature starts monitoring).

        Args:
            iteration: Final iteration
            run_dir: Run directory of the solver
            overwrite: whether to overwrite existing training runs in dir
        """
        snapshot_prefix = self._snapshot_prefix(self.snapshot_save_dir or run_dir)
        self.save_snapshot(iteration, snapshot_prefix + "_final_iter_{}.pth".format(iteration))
        self.save_snapshot(iteration, snapshot_prefix + "_final.pth")

        if dist.is_initialized():
            dist.barrier()

        if self.eval_at_end:
            self.run_eval(run_dir, iteration)

        if self.torchscript_deploy_fn is not None and is_root_process():
            self.torchscript_deploy_fn(self.model, snapshot_prefix + "_final")

        if dist.is_initialized():
            dist.barrier()

        if self.calc_feature_monitoring_stats_func and is_root_process():
            self.calc_feature_monitoring_stats_func(run_dir, overwrite)

        if dist.is_initialized():
            dist.barrier()

    def should_save_snapshot(self, iteration: int, epoch: Optional[int] = None, end_of_epoch: bool = False) -> bool:
        """Determines whether to save a model snapshot at a certain iteration during training.
        Args:
            iteration (int): current iteration
            epoch (Optional[int], optional): Current epoch. This is used if end_of_epoch is True. Defaults to None.
            end_of_epoch (bool, optional): If true, we use epoch to decide wether to save snapshot. Defaults to False.

        Returns:
            bool: True if we want to save snapshot
        """
        if end_of_epoch:
            assert epoch is not None, "end_of_epoch mode, epoch should not be None"
            return (
                epoch != self.epochs and self.snapshot_every_n_epochs != 0 and epoch % self.snapshot_every_n_epochs == 0
            )
        else:
            return (
                iteration != self.start_iter
                and self.snapshot_every_n_iter != 0
                and iteration % self.snapshot_every_n_iter == 0
            )

    def should_run_eval(self, iteration: int, epoch: Optional[int] = None, end_of_epoch: bool = False) -> bool:
        """Determines whether to run evaluation during training

        Args:
            iteration (int): current iteration
            epoch (Optional[int], optional): Current epoch. This is used if end_of_epoch is True. Defaults to None.
            end_of_epoch (bool, optional): If true, we use epoch to decide wether to save snapshot. Defaults to False.

        Returns:
            bool: True if we want to run eval
        """
        if end_of_epoch:
            assert epoch is not None, "end_of_epoch mode, epoch should not be None"
            return epoch != self.epochs and self.eval_every_n_epochs != 0 and epoch % self.eval_every_n_epochs == 0
        else:
            return (
                iteration != self.start_iter and self.eval_every_n_iter != 0 and iteration % self.eval_every_n_iter == 0
            )
    @staticmethod
    def get_best_model_snapshot_path(snapshot_prefix: str) -> str:
        return snapshot_prefix + "_best.pth"

    def execute(
        self,
        gpus: List[int],
        run_dir: str,
        overwrite: bool = False,
        tb_log_dir: Optional[str] = None,
        train_phase: Optional[int] = None,
    ) -> None:
        """
        Main function for Solver.
        Executes `self.iterations` training steps.
        Args:
            gpus: not used (kept for standard interface)
            run_dir: directory of the current training run
            overwrite: whether to overwrite existing training runs in dir
            tb_log_dir: local or s3 directory to write Tensorboard records to
            train_phase: phase of training. only set for multiphase solver
        """
        self.run_train_init_steps(run_dir=run_dir, overwrite=overwrite, tb_log_dir=tb_log_dir)
        snapshot_prefix = self._snapshot_prefix(self.snapshot_save_dir or run_dir)

        cur_epoch = 0
        # Perform time sensitive initialization right before the training loop.
        self.perf_tracker.time_start(TimeStatKey.DATA_LOAD)
        self.perf_tracker.throughput_start(0)
        current_iter = self.start_iter

        if self.profiler_config is not None:
            profiler_config = dataclasses.asdict(self.profiler_config)
            schedule = profiler_config.pop("schedule")
            tb_log_dir = tb_log_dir if tb_log_dir else os.path.join(run_dir, "summary")
            rank = dist.get_rank() if dist.is_initialized() else 0
            prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(**schedule),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(tb_log_dir, worker_name=f"rank_{rank}"),
                **profiler_config,
            )
        else:
            prof = PlaceHolderProfiler()
        prof.start()
        while not self.stopped_early and current_iter < self.iterations:
            for batch in self.train_dataset_loader:
                if current_iter >= self.iterations:
                    break

                self.perf_tracker.time_end(TimeStatKey.DATA_LOAD)

                if self.should_save_snapshot(iteration=current_iter):
                    self.save_snapshot(current_iter, snapshot_prefix + "_iter_{}.pth".format(current_iter))
                if self.should_run_eval(iteration=current_iter):
                    self.run_eval(run_dir, current_iter)

                self.run_train_step(current_iter=current_iter, batch=batch)
                current_iter += 1
                prof.step()

                self.perf_tracker.time_start(TimeStatKey.DATA_LOAD)

        assert (
            self.iterations is None or self.stopped_early or current_iter == self.iterations
        ), f"Something went wrong. Done when current_iter != self.iterations. {current_iter} vs {self.iterations}"
        prof.stop()

        self.run_train_finalize_steps(iteration=current_iter, run_dir=run_dir, overwrite=overwrite)


class BasicSolver(MLEnvSolverBase):
    def __init__(
        self,
        iterations: int = 250000,
        *args,
        **kwargs,
    ):
        """
        Args (in addition to those from MLEnvSolverBase):
            iterations: Num of iterations (an iteration is one gradient update based on a minibatch of examples)

        ```Difference with EpochSolver:
            BasicSolver iterates based on iterations and EpochSolver iterates based on epochs
        """
        # Pass iterations as "steps" to ensure a new solver hash if the number of iterations is changed.
        super().__init__(steps=iterations, iterations=iterations, early_stop_func_check_by_iter=True, *args, **kwargs)


class EpochSolver(MLEnvSolverBase):
    def __init__(
        self,
        epochs: int = 1,
        snapshot_every_n_epochs: int = 0,
        eval_every_n_epochs: int = 0,
        enable_context_join=True,
        divide_by_initial_world_size: bool = True,
        *args,
        **kwargs,
    ):
        """
        Args (in addition to those from MLEnvSolverBase):
            epochs: Number of epochs (full passes through the input dataset)
            snapshot_every_n_epochs: Saves snapshot every n epochs
            eval_every_n_epochs: Evaluate the model every n epochs
            enable_context_join: Whether to enable join context manager See https://pytorch.org/tutorials/advanced/generic_join.html
                for more detail of how this context manager handle uneven inputs on different devices. Default to True for epoch solver.
            divide_by_initial_world_size: Only applies to distributed training, and if Join context manager is enabled.
                If true, at the end of training when fewer than world_size processes still participate in training (because some have finished their
                allocated data shards), the gradients will continue to be divided by world_size. If false, the gradients
                are divided by the remaining number of active processes.
        """
        # Pass epochs as "steps" to ensure a new solver hash if the epoch is changed.
        super().__init__(
            steps=epochs,
            epochs=epochs,
            snapshot_every_n_epochs=snapshot_every_n_epochs,
            eval_every_n_epochs=eval_every_n_epochs,
            early_stop_func_check_by_epoch=True,
            enable_context_join=enable_context_join,
            divide_by_initial_world_size=divide_by_initial_world_size,
            *args,
            **kwargs,
        )


class EvalOnlySolver(Solver):
    """
    A simple solver to run evaluator and optionally register a model.
    """

    def __init__(
        self,
        eval_fn: EvalFunc,
        model: Optional[nn.Module] = None,
        snapshot_filename_prefix: str = "snap",
        snapshot_dir: Optional[str] = None,
        torchscript_deploy_fn: Optional[Callable[[torch.nn.Module, str], None]] = None,
        **kwargs,
    ):
        rank = dist.get_rank() if dist.is_initialized() else 0
        super().__init__(**kwargs, process_rank=rank)
        self.eval_func = eval_fn
        self.model = model
        self.snapshot_filename_prefix = snapshot_filename_prefix
        self.snapshot_save_dir = snapshot_dir
        self.torchscript_deploy_fn = torchscript_deploy_fn

    def _snapshot_prefix(self, save_dir: str) -> str:
        snapshots_dir = os.path.join(save_dir, "snapshots")
        mkdir_p(snapshots_dir)
        return os.path.join(snapshots_dir, self.snapshot_filename_prefix)

    def execute(self, gpus, run_dir, overwrite=False, tb_log_dir=None, train_phase=None) -> None:
        summary_writer = None
        try:
            if is_root_process():
                summary_writer = SummaryWriter(log_dir=tb_log_dir if tb_log_dir else os.path.join(run_dir, "summary"))
            self.eval_func(
                run_dir=run_dir, iteration=0, eval_every_n_iter=0, summary_writer=summary_writer, tb_log_dir=tb_log_dir
            )

            if self.model is not None:
                snapshot_prefix = self._snapshot_prefix(self.snapshot_save_dir or run_dir)
                self.save_snapshot(snapshot_prefix + "_final.pth")
                if self.torchscript_deploy_fn is not None and is_root_process():
                    self.torchscript_deploy_fn(self.model, snapshot_prefix + "_final")
        finally:
            if is_root_process() and summary_writer is not None:
                summary_writer.close()
