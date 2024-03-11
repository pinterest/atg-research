from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import argparse
import json
import logging
import os
import sys
import uuid
from dataclasses import is_dataclass
from os.path import exists
from pydoc import locate
from shutil import ReadError

import hydra
import torch.distributed.elastic.rendezvous.registry as rdzv_registry
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from omnisearchsage.common.trainers.distributed_trainer import PytorchDistributedTrainer
from omnisearchsage.configs.base_configs import resolve
from omnisearchsage.configs.base_configs import to_one_line_json_str
from omnisearchsage.launcher.local_elastic_agent import PinsElasticAgent
from torch.distributed.elastic.agent.server.api import WorkerSpec
from torch.distributed.elastic.multiprocessing import SignalException
from torch.distributed.elastic.multiprocessing import Std
from torch.distributed.elastic.multiprocessing.errors import ChildFailedError
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.utils import parse_rendezvous_endpoint
from torch.distributed.launcher.api import LaunchConfig

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from omnisearchsage.configs.base_configs import ConfigBundle

VALID_MODES = [
    "local",
    "subprocess",
]
DEFAULT_USER = "unknown_user"

LOG = logging.getLogger(__name__)


def update_config_bundle_using_env_vars(config_bundle: ConfigBundle) -> None:
    if "RANK" in os.environ:
        config_bundle.resource_config.node_rank = int(os.environ.get("RANK"))
    if "MASTER_ADDR" in os.environ:
        config_bundle.resource_config.master_addr = os.environ.get("MASTER_ADDR")
    if "MASTER_PORT" in os.environ:
        config_bundle.resource_config.master_port = os.environ.get("MASTER_PORT")


def run_single_process(config_bundle: ConfigBundle) -> None:
    LOG.info("Starting single process...")
    trainer_class = locate(config_bundle.trainer_config.trainer_class)
    if trainer_class is None:
        raise Exception("Failed to find trainer_class at {}".format(config_bundle.trainer_config.trainer_class))
    assert issubclass(
        trainer_class, PytorchDistributedTrainer
    ), f"{config_bundle.trainer_config.trainer_class} is not a PytorchDistributedTrainer"

    trainer = trainer_class(config_bundle=config_bundle)
    trainer.run()


def _get_entrypoint_name(entrypoint: Union[Callable, str, None], args: List[Any]) -> str:
    """Retrive entrypoint name with the rule:
    1. If entrypoint is a function, use ``entrypont.__qualname__``.
    2. If entrypoint is a string, check its value:
        2.1 if entrypoint equals to ``sys.executable`` (like "python"), use the first element from ``args``
            which does not start with hifen letter (for example, "-u" will be skipped).
        2.2 otherwise, use ``entrypoint`` value.
    3. Otherwise, return empty string.
    """
    if isinstance(entrypoint, Callable):  # type: ignore[arg-type]
        return entrypoint.__name__  # type: ignore[union-attr]
    elif isinstance(entrypoint, str):
        if entrypoint == sys.executable:
            return next((arg for arg in args if arg[0] != "-"), "")


def _get_addr_and_port(
    rdzv_parameters: RendezvousParameters,
) -> Tuple[Optional[str], Optional[int]]:
    if rdzv_parameters.backend != "static":
        return None, None
    endpoint = rdzv_parameters.endpoint
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("Endpoint is missing in endpoint. Try to add --master_addr and --master_port")
    master_addr, master_port = parse_rendezvous_endpoint(endpoint, default_port=-1)
    if master_port == -1:
        raise ValueError(f"port is missing in endpoint: {endpoint}. Try to specify --master_port")
    return master_addr, master_port


def run_local_torch_run(config_bundle: ConfigBundle, args: argparse.Namespace = None) -> None:
    update_config_bundle_using_env_vars(config_bundle)

    # Note: here, resource_config.node_rank is populated and safe to use

    if config_bundle.resource_config.nproc_per_node <= 0:
        print("Num processs per node is 0. Training will not run and this is a no-op. ")
        return
    return run_local_torch_run_rdzv(config_bundle, args)


def _assemble_subprocess_cli(config_bundle, args: argparse.Namespace = None) -> List[str]:
    """Assemble the command line arguments for the subprocess."""
    subprocess_args = [
        f"--config_bundle={to_one_line_json_str(config_bundle)}",
        "--mode=subprocess",
    ]

    return [sys.executable, "-u", __file__] + subprocess_args


def run_local_torch_run_rdzv(config_bundle: ConfigBundle, args: argparse.Namespace = None) -> None:
    """Launch train job using pytorch's rendezvous launcher."""
    LOG.info("rdzv_configs is: {}".format(config_bundle.resource_config.rdzv_configs))

    config = LaunchConfig(
        min_nodes=config_bundle.resource_config.num_nodes,
        max_nodes=config_bundle.resource_config.num_nodes,
        nproc_per_node=config_bundle.resource_config.nproc_per_node,
        run_id=uuid.uuid4().hex,
        role="default",
        rdzv_endpoint=f"{config_bundle.resource_config.master_addr}:{config_bundle.resource_config.master_port}",
        rdzv_backend=config_bundle.resource_config.rdzv_backend,
        rdzv_configs=config_bundle.resource_config.rdzv_configs if config_bundle.resource_config.rdzv_configs else {},
        max_restarts=0,
        monitor_interval=5,
        start_method="spawn",
        redirects=Std.from_str("0"),
        tee=Std.from_str("0"),
        log_dir=None,
    )

    rdzv_parameters = RendezvousParameters(
        backend=config.rdzv_backend,
        endpoint=config.rdzv_endpoint,
        run_id=config.run_id,
        min_nodes=config.min_nodes,
        max_nodes=config.max_nodes,
        **config.rdzv_configs,
    )

    if config_bundle.resource_config.rdzv_backend == "static":
        # rdzv_backend=static requires passing the node's rank directly in RendezvousParameters
        rdzv_parameters.config["rank"] = config_bundle.resource_config.node_rank

    master_addr, master_port = _get_addr_and_port(rdzv_parameters)

    LOG.info("LaunchConfig: {} master_addr={} master_port={}".format(config, master_addr, master_port))

    entrypoint, *subprocess_args = _assemble_subprocess_cli(config_bundle, args)

    spec = WorkerSpec(
        role=config.role,
        local_world_size=config.nproc_per_node,
        entrypoint=entrypoint,
        args=tuple(subprocess_args),
        rdzv_handler=rdzv_registry.get_rendezvous_handler(rdzv_parameters),
        max_restarts=config.max_restarts,
        monitor_interval=config.monitor_interval,
        redirects=config.redirects,
        tee=config.tee,
        master_addr=master_addr,
        master_port=master_port,
    )

    agent = PinsElasticAgent(
        spec=spec, start_method=config.start_method, log_dir=config.log_dir, config_bundle=config_bundle
    )

    shutdown_rdzv = True
    try:
        result = agent.run()
        # records that agent.run() has succeeded NOT that workers have succeeded

        if result.is_failed():
            # ChildFailedError is treated specially by @record
            # if the error files for the failed children exist
            # @record will copy the first error (root cause)
            # to the error file of the launcher process.
            raise ChildFailedError(
                name=_get_entrypoint_name(entrypoint, subprocess_args),
                failures=result.failures,
            )
        return result.return_values
    except ChildFailedError:
        raise
    except SignalException:
        # when the agent dies with a signal do NOT shutdown the rdzv_handler
        # since this closes the rendezvous on this rdzv_id permanently and
        # prevents any additional scaling events
        shutdown_rdzv = False
        raise
    except ReadError as e:
        LOG.warning("ReadError: {}".format(e))
    except Exception:
        raise
    finally:
        if shutdown_rdzv:
            spec.rdzv_handler.shutdown()


def parse_config_bundle(config_bundle: str) -> Union[Dict[str, Any], Type[ConfigBundle]]:
    # Validate config_bundle and setup Hydra ConfigStore.
    # 1. check if config_bundle is a json file.
    if exists(config_bundle):
        with open(config_bundle, "r") as f:
            return json.loads(f.read())
    # 2. check if config_bundle is a import path of a ConfigBundle class.
    elif locate(config_bundle):
        return locate(config_bundle)
    else:
        try:
            return json.loads(config_bundle)
        except Exception:
            sys.exit(f"ERROR: unrecognized config_bundle: {config_bundle}. Use --help for more information.")


def launcher_main(config_bundle: ConfigBundle, args: argparse.Namespace):
    """Performs some preprocessing of the input config_bundle, then launches the job.

    Args:
        config_bundle:
        args: See: create_arg_parser()
    Returns:

    """
    LOG.info(OmegaConf.to_yaml(config_bundle))
    if args.mode == "local":
        run_local_torch_run(config_bundle, args)
    elif args.mode == "subprocess":
        run_single_process(config_bundle)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch training.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_bundle",
        type=str,
        required=True,
        help="""Import path of a ConfigBundle or path of a ConfigBundle json file.
            Use "--foo=bar" or "++foo=bar" to override the default values in provided config_bundle.
            See more details at https://hydra.cc/docs/advanced/override_grammar/basic.""",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=VALID_MODES,
        help="""Mode to run the training.
            local: Run the training in a single process.
            subprocess: Run the training in a subprocess.
            """,
    )
    return parser


def main():
    parser = create_arg_parser()

    args, unknown = parser.parse_known_args()

    config_bundle_node = parse_config_bundle(args.config_bundle)
    config_name = "config"
    cs = ConfigStore.instance()
    cs.store(name=config_name, node=config_bundle_node)

    # Hydra doesn't use "--" as arg prefix, so change to "++"
    for i in range(len(unknown)):
        unknown[i] = unknown[i].replace("--", "++")

    # Reset sys.argv which excludes the args for this script defined above.
    # Also disable Hydra's outputs.
    sys.argv = [sys.argv[0]] + unknown + ["hydra.run.dir=/var/log/outputs", "hydra.output_subdir=null"]

    @hydra.main(config_path=None, config_name=config_name, version_base="1.2")
    def hydra_main(config_bundle_dictcfg: DictConfig) -> None:
        # Convert config_bundle from DictConfig to ConfigBundle.
        if is_dataclass(config_bundle_node):
            # If args.config_bundle is ConfigBundle class, convert to object directly.
            config_bundle: ConfigBundle = resolve(config_bundle_dictcfg)
        else:
            # If args.config_bundle is json file, merge it with the ConfigBundle instance.
            config_bundle: ConfigBundle = OmegaConf.create(config_bundle_dictcfg)
        return launcher_main(config_bundle, args)

    hydra_main()


if __name__ == "__main__":
    main()
