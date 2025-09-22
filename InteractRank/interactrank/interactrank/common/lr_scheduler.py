from __future__ import annotations

from typing import Callable
from typing import Optional

import math
import warnings

from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler


def get_constant_schedule(optimizer, last_epoch=-1):
    """Create a schedule with a constant learning rate."""
    custom_repr = "ConstantLRSchedule()"
    return LambdaLRWithRepr(custom_repr, optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    custom_repr = f"ConstantLRScheduleWithWarmup(num_warmup_steps={num_warmup_steps})"
    return LambdaLRWithRepr(custom_repr, optimizer, lr_lambda, last_epoch=last_epoch)


def get_gradual_warmup_with_min_lr(optimizer, num_warmup_steps, min_ratio=0.01, last_epoch=-1):
    """Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between min_ratio and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps * min_ratio:
            return min_ratio
        elif current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    custom_repr = f"WarmupLRSchedule(num_warmup_steps={num_warmup_steps},min_ratio={min_ratio})"
    return LambdaLRWithRepr(custom_repr, optimizer, lr_lambda, last_epoch=last_epoch)


def get_gradual_warmup_schedule(optimizer, num_warmup_steps, base_lr=0.1, target_lr=0.1, last_epoch=-1):
    """Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between `base_lr` and `target_lr`.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            target_diff = target_lr - base_lr
            step_diff = float(current_step) / float(max(1.0, num_warmup_steps)) * target_diff
            return (base_lr + step_diff) / base_lr
        return target_lr / base_lr

    custom_repr = (
        f"GradualWarmupLRSchedule(base_lr={base_lr}, target_lr={target_lr}, num_warmup_steps={num_warmup_steps})"
    )
    return LambdaLRWithRepr(custom_repr, optimizer, lr_lambda, last_epoch=last_epoch)


def get_gradual_decay_schedule(optimizer, total_steps, base_lr=0.1, end_lr=1e-5, last_epoch=-1):
    """Create a schedule where learning rate is linearly decayed from `base_lr` to `end_lr`."""

    def lr_lambda(current_step):
        progress = float(current_step) / float(total_steps)
        current_lambda = (end_lr + (base_lr - end_lr) * (1.0 - progress)) / base_lr
        min_lambda = end_lr / base_lr  # defensive but not strictly necessary
        return max(min_lambda, current_lambda)

    custom_repr = f"GradualDecayLRSchedule(base_lr={base_lr}, end_lr={end_lr}, total_steps={total_steps})"
    return LambdaLRWithRepr(custom_repr, optimizer, lr_lambda, last_epoch=last_epoch)


def get_exponential_decay_schedule(optimizer, half_life_steps, last_epoch=-1):
    def lr_lambda(current_step):
        return 2 ** (-current_step / half_life_steps)

    custom_repr = f"ExponentialDecayLRSchedule(half_life_steps={half_life_steps})"
    return LambdaLRWithRepr(custom_repr, optimizer, lr_lambda, last_epoch=last_epoch)


class CosineAnnealingLRScheduler(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(T_max={self.T_max}, eta_min={self.eta_min})"


class CosineAnnealingWarmRestartsScheduler(CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0.0, last_epoch=-1):
        super().__init__(optimizer, T_0, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(T_0={self.T_0}, T_mult={self.T_mult}, eta_min={self.eta_min})"


class StepLRScheduler(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, last_epoch=-1):
        super(StepLRScheduler, self).__init__(optimizer, step_size, gamma=gamma, last_epoch=last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma={self.gamma}, step_size={self.step_size})"


class MultiStepLRScheduler(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        super(MultiStepLRScheduler, self).__init__(optimizer, milestones=milestones, gamma=gamma, last_epoch=last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(gamma={self.gamma}, milestones={sorted(self.milestones.keys())})"


class LambdaLRWithRepr(LambdaLR):
    def __init__(self, custom_repr: str, optimizer: Optimizer, lr_lambda: Callable[[int], float], last_epoch=-1):
        super().__init__(optimizer, lr_lambda, last_epoch)
        self.custom_repr = custom_repr

    def __repr__(self):
        return f"{self.custom_repr}"


class GradualWarmupCompositeLRScheduler(_LRScheduler):
    """Gradually warm-up (increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Optionally composite with another LR scheduler, i.e., use `after_scheduler` after warmup phase.

    Note that the `after_scheduler` references the multiplier-adjusted base LRs of the optimizer. Therefore, if using
    the scheduler returned by `get_gradual_decay_schedule`, ensure that `base_lr` is set correctly.

    This composite scheduler does not support discontinuous warmup schedules (e.g. warmup the learning rate from
    0.0 to 2.0, then gradually decrease the learning rate from 1.0 to 0.0), however this is uncommon in practice.

    Adapted from open-source implementation: https://github.com/ildoonet/pytorch-gradual-warmup-lr

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_warmup_steps: target learning rate is reached at num_warmup_steps, gradually.
        after_scheduler_fn: after num_warmup_steps, use scheduler returned by this function (eg. ReduceLROnPlateau).
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0.
            if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        num_warmup_steps: int,
        after_scheduler_fn: Optional[Callable[[Optimizer], _LRScheduler]] = None,
        multiplier=1.0,
    ):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater than or equal to 1.")
        self.num_warmup_steps = num_warmup_steps
        self.after_scheduler_fn = after_scheduler_fn
        self.after_scheduler = None
        self.finished = False
        super(GradualWarmupCompositeLRScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.num_warmup_steps:
            if self.after_scheduler_fn:
                if not self.finished:
                    self.after_scheduler = self.after_scheduler_fn(self.optimizer)
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.num_warmup_steps) for base_lr in self.base_lrs]
        else:
            return [
                base_lr * ((self.multiplier - 1.0) * self.last_epoch / self.num_warmup_steps + 1.0)
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.num_warmup_steps)
            self._last_lr = self.after_scheduler.get_last_lr()
        else:
            return super(GradualWarmupCompositeLRScheduler, self).step(epoch)

    def __repr__(self):
        if self.after_scheduler_fn:
            return f"{self.__class__.__name__}(multiplier={self.multiplier}, num_warmup_steps={self.num_warmup_steps}, after_scheduler_fn={self.after_scheduler_fn})"
        else:
            return f"{self.__class__.__name__}(multiplier={self.multiplier}, num_warmup_steps={self.num_warmup_steps})"


class OneCycleLRScheduler(_LRScheduler):
    r"""
    This scheduler was adapted from `torch.optim.lr_scheduler.OneCycleLR`.

    This version supports per-phase annealing strategies and inverse square root annealing.

    For simplicity, this version does not support the three-phase policy.

    Sets the learning rate of each parameter group according to the
    1cycle learning rate policy. The 1cycle policy anneals the learning
    rate from an initial learning rate to some maximum learning rate and then
    from that maximum learning rate to some minimum learning rate much lower
    than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        pct_start (float): The percentage of the cycle (in number of steps) spent
            increasing the learning rate.
            Default: 0.3
        phase1_anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy for phase 1: "cos" for cosine annealing, "linear" for linear annealing.
            Default: 'linear'
        phase2_anneal_strategy (str): {'cos', 'linear', 'rsqrt'}
            Specifies the annealing strategy for phase 2: "cos" for cosine annealing, "linear" for linear annealing,
            "rsqrt" for reciprocal square root annealing.
            Default: 'cos'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.85
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.95
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        timescale (int): Determines the rate of inverse square root annealing.
            Default: 10_000
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """

    def __init__(
        self,
        optimizer,
        max_lr,
        total_steps=None,
        epochs=None,
        steps_per_epoch=None,
        pct_start=0.3,
        phase1_anneal_strategy="linear",
        phase2_anneal_strategy="cos",
        cycle_momentum=True,
        base_momentum=0.85,
        max_momentum=0.95,
        div_factor=25.0,
        final_div_factor=1e4,
        timescale=10_000,
        last_epoch=-1,
        verbose=False,
    ):
        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError("{} is not an Optimizer".format(type(optimizer).__name__))
        self.optimizer = optimizer

        # Validate total_steps
        if total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected positive integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected positive integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch

        self._schedule_phases = [
            {
                "end_step": float(pct_start * self.total_steps) - 1,
                "start_lr": "initial_lr",
                "end_lr": "max_lr",
                "start_momentum": "max_momentum",
                "end_momentum": "base_momentum",
            },
            {
                "end_step": self.total_steps - 1,
                "start_lr": "max_lr",
                "end_lr": "min_lr",
                "start_momentum": "base_momentum",
                "end_momentum": "max_momentum",
            },
        ]

        # Validate pct_start
        if pct_start < 0 or pct_start > 1 or not isinstance(pct_start, float):
            raise ValueError("Expected float between 0 and 1 pct_start, but got {}".format(pct_start))
        self.warmup_steps = float(pct_start * self.total_steps)
        self.timescale = timescale
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        # Validate anneal_strategy
        self.anneal_funcs = []

        if phase1_anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                f"phase1_anneal_strategy must be one of 'cos' or 'linear', instead got {phase1_anneal_strategy}"
            )
        elif phase1_anneal_strategy == "cos":
            self.anneal_funcs.append(self._annealing_cos)
        elif phase1_anneal_strategy == "linear":
            self.anneal_funcs.append(self._annealing_linear)

        if phase2_anneal_strategy not in ["cos", "linear", "rsqrt"]:
            raise ValueError(
                f"phase2_anneal_strategy must be one of 'cos', 'linear', or 'rsqrt', instead got {phase2_anneal_strategy}"
            )
        elif phase2_anneal_strategy == "cos":
            self.anneal_funcs.append(self._annealing_cos)
        elif phase2_anneal_strategy == "linear":
            self.anneal_funcs.append(self._annealing_linear)
        elif phase2_anneal_strategy == "rsqrt":
            self.anneal_funcs.append(self._annealing_rsqrt)

        # Initialize learning rate variables
        max_lrs = self._format_param("max_lr", self.optimizer, max_lr)
        if last_epoch == -1:
            for idx, group in enumerate(self.optimizer.param_groups):
                group["initial_lr"] = max_lrs[idx] / div_factor
                group["max_lr"] = max_lrs[idx]
                group["min_lr"] = group["initial_lr"] / final_div_factor

        # Initialize momentum variables
        self.cycle_momentum = cycle_momentum
        if self.cycle_momentum:
            if "momentum" not in self.optimizer.defaults and "betas" not in self.optimizer.defaults:
                raise ValueError("optimizer must support momentum with `cycle_momentum` option enabled")
            self.use_beta1 = "betas" in self.optimizer.defaults
            max_momentums = self._format_param("max_momentum", optimizer, max_momentum)
            base_momentums = self._format_param("base_momentum", optimizer, base_momentum)
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(max_momentums, base_momentums, optimizer.param_groups):
                    if self.use_beta1:
                        group["betas"] = (m_momentum, *group["betas"][1:])
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        super().__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    "expected {} values for {}, got {}".format(len(optimizer.param_groups), name, len(param))
                )
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end - start) * pct + start

    def _annealing_rsqrt(self, start, end, pct):
        "Reciprocal square root anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        current_step = self.warmup_steps + pct * (self.total_steps - self.warmup_steps)
        shift = self.timescale - self.warmup_steps
        decay = 1.0 / math.sqrt((current_step + shift) / self.timescale)
        return end + (start - end) * decay

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, please use `get_last_lr()`.", UserWarning
            )

        lrs = []
        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}".format(step_num, self.total_steps)
            )

        for group in self.optimizer.param_groups:
            start_step = 0
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    start_lr = group[phase["start_lr"]]
                    end_lr = group[phase["end_lr"]]
                    computed_lr = self.anneal_funcs[i](start_lr, end_lr, pct)
                    if self.cycle_momentum:
                        start_momentum = group[phase["start_momentum"]]
                        end_momentum = group[phase["end_momentum"]]
                        computed_momentum = self.anneal_funcs[i](start_momentum, end_momentum, pct)
                    break
                start_step = phase["end_step"]

            lrs.append(computed_lr)
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (computed_momentum, *group["betas"][1:])
                else:
                    group["momentum"] = computed_momentum

        return lrs

    def __repr__(self):
        return f"{self.__class__.__name__}(timescale={self.timescale}, warmup_steps={int(self.warmup_steps)}, total_steps={self.total_steps})"
