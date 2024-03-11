from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import copy
import json
import operator
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime

from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

TORCH_RDZV_DEFAULT_JOIN_TIMEOUT_SECS = 7200  # 2 hours. in seconds
OmegaConf.register_new_resolver("multiply", operator.mul, replace=True)


def default(value: Any):
    """Returns a dataclass field with its default value set to the input value.
    This takes care to ensure that each dataclass instance is instantiated with its own `value` instance, to avoid
    unexpected aliasing issues.
    Implementation note: to ensure that each dataclass instance works with its own `value` instance, this creates a
    deep-copy of the input `value`.
    Convenience function for working with dataclasses.

    Args:
        value: Default value. Must be copy-able via `copy.deepcopy`.

    Returns:
        field:

    """
    # Tricky: to ensure that subsequent external modifications to `value` don't leak to subsequent dataclass
    # instantiations, we create an inner copy here.
    # The second copy in the lambda is necessary so that each dataclass instance gets its own `value` instance
    value_copy = copy.deepcopy(value)
    return field(default_factory=lambda: copy.deepcopy(value_copy))


DEFAULT_RDZV_CONFIG = {
    # For multi-node parallel jobs (where num_nodes>1), the default 600 secs (10 mins) is sometimes not enough, as
    # we will sometimes hit this limit due to the time required to spin up and initialize instances.
    # Thus, we substantially increase it here.
    "join_timeout": TORCH_RDZV_DEFAULT_JOIN_TIMEOUT_SECS,
    # Only for TCPStore. In seconds.
    "read_timeout": 1800,
}


@dataclass
class ResourceConfig:
    """
    Machine resource configuration.
    """

    # The number of nodes to use for training.
    num_nodes: int = 1
    # Number of GPUs per node.
    gpus_per_node: int = 1
    # Number of CPUs to request per node.
    cpus_per_node: int = 1
    # System memory in GiB to request per node.
    mem_gb_per_node: int = 32
    # The number of processes to launch on each node. By default set to gpus_per_node so that each
    # process can be bound to a single GPU.
    nproc_per_node: int = "${resource_config.gpus_per_node}"
    # Training world size.
    world_size: int = "${multiply:${resource_config.num_nodes},${resource_config.nproc_per_node}}"
    # Master node (node rank 0)'s address that needs to be used for communciation during
    # distributed training.
    master_addr: str = "127.0.0.1"
    # Master node (node rank 0)'s free port that needs to be used for communciation during
    # distributed training.
    master_port: str = "29500"

    # For pytorch rendezvous.
    # Important params:
    # "join_timeout": How long to wait for all hosts to register before giving up. In seconds.
    # For more info:
    #   https://pytorch.org/docs/stable/elastic/rendezvous.html#torch.distributed.elastic.rendezvous.dynamic_rendezvous.create_handler  # noqa
    rdzv_configs: Optional[Dict] = default(DEFAULT_RDZV_CONFIG)

    # pytorch rendezvous backend to use.
    # Typical choices: static, c10d
    rdzv_backend: str = "c10d"

    # ====== Do not override the following fields ====== #
    # A list of GPUs to use for one process.
    gpus: Optional[List[int]] = None
    # The rank of the node for multi-node distributed.
    node_rank: int = 0

    def __post_init__(self):
        assert self.__class__ == ResourceConfig, "Inheriting ResourceConfig is not allowed."


def to_one_line_json_str(config_bundle: Union[ConfigBundle, DictConfig]) -> str:
    """
    Convert config bundle to one line json string.

    Arguments:
        config_bundle {Union[ConfigBundle, DictConfig]} -- input config bundle.
    """
    config_bundle_dict = (
        asdict(config_bundle) if isinstance(config_bundle, ConfigBundle) else OmegaConf.to_container(config_bundle)
    )
    config_bundle_json_str = json.dumps(config_bundle_dict, separators=(',', ':'))

    # Make sure the json str can convert back to the same dict. For example, using integer as dict key is not supported.
    assert config_bundle_dict == json.loads(
        config_bundle_json_str
    ), f"Make sure your ConfigBundle is jsonable!! \nBefore converting:{config_bundle_dict}\nAfter converting:{json.loads(config_bundle_json_str)}"

    return config_bundle_json_str


@dataclass
class TrainerConfig:
    # Import path to DistributedTrainer.
    trainer_class: str = "missing"
    # Owner of this model.
    user: Optional[str] = None

    # Namespace
    namespace: str = "missing"

    # NCCL timeout in minutes.
    nccl_timeout_mins: int = 30

    # S3 Bucket for output storage
    s3_bucket: Optional[str] = None

    # ===== Output Configs ===== #
    # S3 base directory to store train outputs to, after training is successfully complete.
    s3_save_dir: str = (
        "s3://${trainer_config.s3_bucket}/training/cli/${trainer_config.user}/${trainer_config.namespace}/"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    # Locally directory to store tensorboard outputs.
    tb_log_dir: Optional[str] = "/var/log/tf_logs"
    # If true, upload training output to S3 after training finishes successfully.
    persist_to_s3: bool = True

    def __post_init__(self):
        assert self.__class__ == TrainerConfig, "Inheriting TrainerConfig is not allowed."


def resolve(config_bundle: Union[ConfigBundle, DictConfig]) -> Union[ConfigBundle, DictConfig]:
    """
    Resolve the variable interpolation in the ConfigBundle/DictConfig and returns a ConfigBundle/DictConfig
    respectively.

    See more details at: https://omegaconf.readthedocs.io/en/2.1_branch/usage.html#variable-interpolation
    """
    return OmegaConf.to_object(OmegaConf.create(config_bundle))


@dataclass
class ConfigBundle:
    trainer_config: TrainerConfig = default(TrainerConfig())
    resource_config: ResourceConfig = default(ResourceConfig())

    # ====== Do not override the following fields ====== #
    config_bundle_class: Optional[str] = None
