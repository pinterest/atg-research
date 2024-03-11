from __future__ import annotations

from typing import List
from typing import Optional

from torch.distributed import get_rank
from torch.distributed import is_initialized


class Solver(object):
    def __init__(self, **kwargs) -> None:
        pass

    def execute(
        self,
        gpus: List[int],
        run_dir: str,
        overwrite: bool = False,
        tb_log_dir: Optional[str] = None,
        train_phase: Optional[int] = None,
    ):
        """Since different deep learning framework has different execution
        functions, which should be called here.

        Args:
            gpus: a list of gpus
            run_dir: working directory where logs, evaluation results, etc should be stored for this specific
                     training run
            overwrite: whether to allow overwriting the previous run.
            tb_log_dir: local or s3 directory to write Tensorboard records to.
            train_phase: phase of training (if using multiphase solver)
        """
        raise NotImplementedError()


def is_root_process():
    """
    Any non-distributed training is considered a root process
    For distributed training, rank == 0 is the root process
    """
    return not (is_initialized() and get_rank() != 0)
