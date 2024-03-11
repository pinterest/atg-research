from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import os
import shutil

from torch.distributed.elastic.agent.server.local_elastic_agent import LocalElasticAgent
from torch.distributed.elastic.metrics.api import prof
from torch.distributed.elastic.multiprocessing import start_processes

if TYPE_CHECKING:
    from torch.distributed.elastic.agent.server.api import WorkerGroup
    from torch.distributed.elastic.agent.server.api import WorkerSpec

    from ml_resources.configs.base_configs import ConfigBundle


class PinsElasticAgent(LocalElasticAgent):
    """This inherits the Torch LocalElasticAgent and the gpu device_ids and the mlflow run id.
    :param LocalElasticAgent: _description_
    """

    def __init__(
        self,
        spec: WorkerSpec,
        config_bundle: ConfigBundle,
        start_method: str = "spawn",
        exit_barrier_timeout: float = 300,
        log_dir: Optional[str] = None,
    ) -> None:
        self._config_bundle = config_bundle
        super().__init__(spec, start_method, exit_barrier_timeout, log_dir)

    @prof
    def _start_workers(self, worker_group: WorkerGroup) -> Dict[int, Any]:
        spec = worker_group.spec
        store = worker_group.store
        assert store is not None
        master_addr, master_port = super()._get_master_addr_port(store)

        # We need to encode static IPS here using env if we are using rdvz_backend as static
        # master_addr = os.environ["MASTER_ADDR"]

        restart_count = spec.max_restarts - self._remaining_restarts

        use_agent_store = spec.rdzv_handler.get_backend() == "static"

        args: Dict[int, Tuple] = {}
        envs: Dict[int, Dict[str, str]] = {}
        for worker in worker_group.workers:
            local_rank = worker.local_rank
            worker_env = {
                "LOCAL_RANK": str(local_rank),
                "RANK": str(worker.global_rank),
                "GROUP_RANK": str(worker_group.group_rank),
                "ROLE_RANK": str(worker.role_rank),
                "ROLE_NAME": spec.role,
                "LOCAL_WORLD_SIZE": str(spec.local_world_size),
                "WORLD_SIZE": str(worker.world_size),
                "GROUP_WORLD_SIZE": str(worker_group.group_world_size),
                "ROLE_WORLD_SIZE": str(worker.role_world_size),
                "MASTER_ADDR": master_addr,
                "MASTER_PORT": str(master_port),
                "TORCHELASTIC_RESTART_COUNT": str(restart_count),
                "TORCHELASTIC_MAX_RESTARTS": str(spec.max_restarts),
                "TORCHELASTIC_RUN_ID": spec.rdzv_handler.get_run_id(),
                "TORCHELASTIC_USE_AGENT_STORE": str(use_agent_store),
                "NCCL_ASYNC_ERROR_HANDLING": os.getenv("NCCL_ASYNC_ERROR_HANDLING", str(1)),
                "NCCL_DEBUG": os.getenv("NCCL_DEBUG", "INFO"),
                # "PYTHONPATH": os.environ["PYTHONPATH"] + ":/machine-learning",
            }
            if "OMP_NUM_THREADS" in os.environ:
                worker_env["OMP_NUM_THREADS"] = os.environ["OMP_NUM_THREADS"]

            envs[local_rank] = worker_env
            worker_args = list(spec.args)
            if self._config_bundle.resource_config.gpus_per_node > 0:
                worker_args.append(f"--resource_config.gpus=[{local_rank}]")
            args[local_rank] = tuple(worker_args)

        # scaling events do not count towards restarts (gets same attempt #)
        # remove existing log dir if this restart is due to a scaling event
        attempt_log_dir = os.path.join(self._log_dir, f"attempt_{restart_count}")
        shutil.rmtree(attempt_log_dir, ignore_errors=True)
        os.makedirs(attempt_log_dir)

        assert spec.entrypoint is not None
        self._pcontext = start_processes(
            name=spec.role,
            entrypoint=spec.entrypoint,
            args=args,
            envs=envs,
            log_dir=attempt_log_dir,
            start_method=self._start_method,
            redirects=spec.redirects,
            tee=spec.tee,
        )

        return self._pcontext.pids()
