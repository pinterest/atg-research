from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Optional

from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import _LRScheduler

if TYPE_CHECKING:
    from torch.optim import Optimizer


def get_constant_schedule(optimizer, last_epoch=-1):
    """Create a schedule with a constant learning rate."""
    custom_repr = "ConstantLRSchedule()"
    return LambdaLRWithRepr(custom_repr, optimizer, lambda _: 1, last_epoch=last_epoch)


class LambdaLRWithRepr(LambdaLR):
    def __init__(self, custom_repr: str, optimizer: Optimizer, lr_lambda: Callable[[int], float], last_epoch=-1):
        super().__init__(optimizer, lr_lambda, last_epoch)
        self.custom_repr = custom_repr

    def __repr__(self):
        return f"{self.custom_repr}"


class CosineAnnealingLRScheduler(CosineAnnealingLR):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        super().__init__(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(T_max={self.T_max}, eta_min={self.eta_min})"


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
            ]  # noqa: E501

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
            return f"{self.__class__.__name__}(multiplier={self.multiplier}, num_warmup_steps={self.num_warmup_steps}, after_scheduler_fn={self.after_scheduler_fn})"  # noqa: E501
        else:
            return f"{self.__class__.__name__}(multiplier={self.multiplier}, num_warmup_steps={self.num_warmup_steps})"
