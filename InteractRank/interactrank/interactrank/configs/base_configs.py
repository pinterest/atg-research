

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union
from typing import Optional
from enum import Enum

import enum
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

from typing import Dict
from interactrank.common.types import UtilityKey
from interactrank.common.types import SearchHeadNames
from interactrank.common.types import SearchLabels

ENGAGEMENT_TASK = "engagement"
RELEVANCE_TASK = "relevance"
TASKS = [ENGAGEMENT_TASK, RELEVANCE_TASK]
TCP_JOB_KILL_TTL_4_DAYS = 60 * 60 * 24 * 4
DEFAULT_TABULAR_LONG_CLICK_WEIGHT_LWS = 20.0

class Metric(enum.Enum):
    LOG_LOSS = "log_loss"
    PR_AUC = "pr_auc"
    ROC_AUC = "roc_auc"
    ACTUAL_EVENTS = "actual_events"
    EXPECTED_EVENTS = "expected_events"
    CALIBRATION = "calibration"
    LABEL_POS_SUM = "positive_label_sum"
    LABEL_POS_RATE = "positive_label_rate"
    WEIGHT_SUM = "weight_sum"
    POS_WEIGHT_SUM = "pos_weight_sum"
    LABEL_NEG_SUM = "negative_label_sum"


class HeadConfig:
    def __init__(
        self,
        name: SearchHeadNames,
        utility_key: UtilityKey,
        label: SearchLabels,
        label_weight: float,
        index_in_prediction: int,
    ):
        """
        :param name: Name of this label
        :param utility_key: Corresponding UtilityKey of this head,
                            used when storing the utility weight for this head in serving
        :param label_weight: Weight for this head in the loss function
        :param index_in_prediction: Index of this head in `predictions` output in forward pass during training
        """
        self.name = name
        self.utility_key = utility_key
        self.label = label
        self.label_weight = label_weight
        self.index_in_prediction = index_in_prediction

SEARCH_LW_HEAD_CONFIGS = [
    # **NOTE**: DO NOT CHANGE THE ORDER OF THE HEADS IN THIS LIST;
    # THE ORDER IS IMPORTANT FOR FETCHING THE HEAD SCORES AND DERIVING LABELS AND WEIGHTS.
    HeadConfig(
        name=SearchHeadNames.ENGAGEMENT,
        utility_key=UtilityKey.ENGAGEMENT,
        label=SearchLabels.ENGAGEMENT,
        # The weight of the binary engagement head was set the same as that of the label
        # with the highest weight, i.e., shortclick: 20.0
        label_weight=20.0,
        index_in_prediction=0,
    ),
HeadConfig(
        name=SearchHeadNames.REPIN,
        utility_key=UtilityKey.REPIN,
        label=SearchLabels.REPIN,
        label_weight=7.0,
        index_in_prediction=1,
    ),
    HeadConfig(
        name=SearchHeadNames.LONG_CLICK,
        utility_key=UtilityKey.LONG_CLICK,
        label=SearchLabels.LONG_CLICK,
        label_weight=20.0,
        index_in_prediction=2,
    ),
]

ENGAGEMENT_HEAD = SearchHeadNames.ENGAGEMENT.value
NUM_HEADS = len(SEARCH_LW_HEAD_CONFIGS)

LABEL_FIELD = "binary_label"
WEIGHT_FIELD = "weight"
LABEL_TYPE = "label_type"
PREDICTIONS_FIELD = "predictions"

DEFAULT_DEVICE_LIST = ("cpu", "cuda")
SOFTMAX_LOSS = "softmax_loss"
ENG_LOSS = "eng_loss"
LOGGED_RELEVANCE_LOSS = "logged_rel_loss"
TOTAL_LOSS = "total_loss"
PR_AUC = "pr_auc"
ROC_AUC = "roc_auc"

DOT_PRODUCT_FEATURE_FIELD = "84/two_tower_dot_product"
MODEL_CONFIG_JSON = {"preprocessor": 2, "torch_module_info_extra_file": "module_info.json"}
TFRECORD_LABEL_COLUMNS = {LABEL_FIELD, WEIGHT_FIELD, LABEL_TYPE, "user_id"}

QUERY = 'query'
PIN = 'pin'
SEARCHSAGE_VERSION_TO_FEATURE = {
    'v3b': {
        QUERY: '81/common.query.searchSageV3BetaQueryTensorEmbedding',
        PIN: '82/common.pin.searchsage_v3beta_pin_embedding',
    },
    'v3a': {
        QUERY: '81/common.query.searchSageV3AlphaQueryTensorEmbedding',
        PIN: '82/common.pin.searchsage_v3alpha_pin_embedding',
    },
}

NUM_HEADS = len(SEARCH_LW_HEAD_CONFIGS)
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
    persist_to_s3: bool = False

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
