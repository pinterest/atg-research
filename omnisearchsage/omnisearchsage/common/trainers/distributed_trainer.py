from __future__ import annotations

from typing import TYPE_CHECKING
from typing import List
from typing import Optional
from typing import Union

import datetime
import logging
import os
import shutil
import subprocess
import traceback

import torch.distributed
from omnisearchsage.common.solver.solver import is_root_process
from omnisearchsage.common.utils.fs_utils import hash_str
from omnisearchsage.common.utils.fs_utils import mkdir_p

if TYPE_CHECKING:
    from omnisearchsage.configs.base_configs import ConfigBundle
    from trainer.utils.solver import Solver

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)
LOGGER = logging.getLogger(__name__)


class PytorchDistributedTrainer:
    """An abstraction of generic trainer"""

    DEFAULT_TRAIN_OUTPUTS_DIR = "/data1/train_outputs/"

    DEFAULT_NUM_GPUS = 1  # Number of GPUs per process

    def __init__(
        self,
        config_bundle: ConfigBundle,
    ):
        """
        Args:
            config_bundle. The ConfigBundle with the configurations of the training job and UniversalTrainer
                behavior. This is provided if the distributed training is launched from launcher v2.
        """
        # Always initialize config bundle for backward compatibility since some trainer expect this property be set to
        # the UniversalTrainer object when they pass in config bundle to super().__init__()
        self.config_bundle = config_bundle

        self._init_trainer_properties(
            gpus=config_bundle.resource_config.gpus or list(range(self.DEFAULT_NUM_GPUS)),
            namespace=config_bundle.trainer_config.namespace,
            s3_save_dir=config_bundle.trainer_config.s3_save_dir,
            tb_log_dir=config_bundle.trainer_config.tb_log_dir,
            disable_persist_to_s3=not config_bundle.trainer_config.persist_to_s3,
        )
        self._validate_gpu_setting()

    def _validate_gpu_setting(self) -> None:
        # Validating GPU setting
        if self.num_gpus is not None:
            assert len(self._gpus) == self.num_gpus, "Required to provide {} gpus".format(self.num_gpus)

    def _init_trainer_properties(
        self,
        gpus: Optional[List[Union[int, str]]] = None,
        namespace: str = "",
        s3_save_dir: str = "",
        tb_log_dir: str = "",
        disable_persist_to_s3: bool = False,
    ) -> None:
        """Set up properties for this UniversalTrainer object.

        Args:
            gpus: GPU device literal for the GPU worker of this UniversalTrainer object.
            namespace: Namespace for this Trainer. This controls things like: output directory name, mlflow experiment
                id. Typically this will be some function (eg str-repr hash) of the model hyperparameters.
            s3_save_dir: S3 base directory to store train outputs to, after training is successfully complete.
            tb_log_dir: Localy directory to store tensorboard outputs.
            disable_persist_to_s3: If True, this disables the post-training S3 upload functionality.
                Useful for hosts/infra that can't connect to Pinterest S3, eg GCP hosts.

        Returns:
            None
        """
        self._gpus = gpus
        self.namespace = namespace
        self.s3_save_dir = s3_save_dir
        self.tb_log_dir = tb_log_dir
        self.disable_persist_to_s3 = disable_persist_to_s3

    @property
    def base_dir(self) -> str:
        return os.path.join(self.DEFAULT_TRAIN_OUTPUTS_DIR, self.namespace)

    @property
    def gpus(self) -> List[int]:
        """
        Returns: The gpu devices ids
        """
        return self._gpus

    @property
    def device(self):
        return torch.device("cuda") if self.gpus and torch.cuda.is_available() else torch.device("cpu")

    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    @property
    def rank(self) -> int:
        return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

    def create_solver(self) -> Solver:
        """Overwrite this function to implement different solver generator.

        Returns:
            a list of solvers
        """
        raise NotImplementedError()

    def init_torch_distributed(self):
        timeout = datetime.timedelta(minutes=self.config_bundle.trainer_config.nccl_timeout_mins)
        world_size = self.config_bundle.resource_config.world_size

        if self.gpus is None or len(self.gpus) <= 0:
            backend = torch.distributed.Backend.GLOO
        else:
            assert self.num_gpus == 1, "Distributed Training with > 1 GPU per process is not supported yet"
            # NCCL does not respect CUDA_VISIBLE_DEVICES so must change default device this way.
            # This just makes sure the current process will use the GPU given
            backend = torch.distributed.Backend.NCCL
            torch.cuda.set_device(self.gpus[0])

        torch.distributed.init_process_group(
            backend=backend, init_method="env://", world_size=world_size, timeout=timeout
        )

        print(f"Hello from process {self.rank} (out of {self.world_size})!")

    def run(self, base_dir: str = None):
        # Store output into different rank directories
        # distributed init initialization
        # This guard is needed because of how Lego is initializing the trainer. Lego needs to move
        # distributed_init to its run method.
        if not torch.distributed.is_initialized():
            self.init_torch_distributed()

        base_dir = self.base_dir if base_dir is None else base_dir

        base_dir = os.path.join(base_dir, f"rank_{self.rank}")
        print(f"The base dir is {base_dir} for rank {self.rank}")

        mkdir_p(base_dir)
        solver = self.create_solver()

        solver_str = str(solver)
        solver_hash = hash_str(solver_str)
        run_dir = os.path.join(base_dir, solver_hash)
        print("Running directory: {}".format(run_dir))

        mkdir_p(run_dir)
        try:
            trainer_name = self.__class__.__name__.lower()
            with open(os.path.join(run_dir, trainer_name + "_spec.json"), "w") as spec_file:
                spec_file.write(solver_str)

            tb_log_dir = os.path.join(self.tb_log_dir, solver_hash, "summary") if self.tb_log_dir else None
            solver.execute(gpus=self.gpus, run_dir=run_dir, tb_log_dir=tb_log_dir)

        except Exception as e:
            traceback.print_exc()
            raise e
        finally:
            if self.s3_save_dir and not self.disable_persist_to_s3:
                self.persist_to_s3(solver_hash, run_dir, self.s3_save_dir)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def persist_to_s3(self, solver_hash: str, run_dir: str, s3_save_dir: str) -> None:
        # Persist training results (checkpoints, eval results, logs, etc) to s3
        LOGGER.info("Persisting train outputs to S3: {}, {}, {}".format(solver_hash, run_dir, self.s3_save_dir))
        if self.tb_log_dir:
            try:
                # Copy tensorboard records to directory uploaded to s3
                shutil.copytree(os.path.join(self.tb_log_dir, solver_hash, "summary"), os.path.join(run_dir, "summary"))
            except OSError as e:
                print(
                    "WARNING: Couldn't copy Tensorboard records from {} to {}, due to: "
                    "{}".format(
                        os.path.join(self.tb_log_dir, solver_hash, "summary"), os.path.join(run_dir, "summary"), e
                    )
                )

        upload_to_s3_command = f"aws s3 cp --recursive --only-show-errors {run_dir} {s3_save_dir}"
        subprocess.check_call(upload_to_s3_command, shell=True, executable="/bin/bash")
        print("Uploaded training results to: {}".format(s3_save_dir))

    def is_root_process(self) -> bool:
        """
        Override this if you use a distributed trainer
        """
        return is_root_process()

    @property
    def num_gpus(self) -> int:
        """
        Returns: The number of gpus
        """
        return 1

    # =============== Public methods Section Ends ============================== #
