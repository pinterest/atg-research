from __future__ import annotations

from typing import Optional

import os
from dataclasses import dataclass

from omnisearchsage.configs.base_configs import ConfigBundle
from omnisearchsage.configs.base_configs import ResourceConfig
from omnisearchsage.configs.base_configs import TrainerConfig

NUM_GPUS_PER_NODE = 8
CPUS_PER_GPU = 11
MEM_GB_PER_GPU = 130
DEFAULT_TRAIN_RESOURCE_CONFIG = ResourceConfig(
    num_nodes=1,
    gpus_per_node=NUM_GPUS_PER_NODE,
    cpus_per_node=NUM_GPUS_PER_NODE * CPUS_PER_GPU,
    mem_gb_per_node=NUM_GPUS_PER_NODE * MEM_GB_PER_GPU,
)

DEFAULT_EVAL_RESOURCE_CONFIG = ResourceConfig(
    num_nodes=1,
    gpus_per_node=1,
    cpus_per_node=11,
    mem_gb_per_node=MEM_GB_PER_GPU,
)


@dataclass
class OmniSearchSageAppConfig:
    """
    Default hyperparameters for OmniSearchSage model
    """

    batch_size: int = 1024 * 10
    neg_ratio: int = 4
    warmup_steps: int = 750
    log10_base_lr: float = -2.7
    iterations: int = 150_000
    eval_index_size: str = "10M"
    train_index_size: str = "50M"
    num_workers: int = 3
    eval_every_n_iter: int = 25_000
    max_grad_norm: Optional[float] = None
    query_base_model_name: str = "distilbert-base-multilingual-cased"


USER = os.getenv("USER", "")


@dataclass
class OmniSearchSageTrainingConfigBundle(ConfigBundle):
    """
    ConfigBundle for OmniSearchsage model training

    python3.8 omnisearchsage/launcher/launcher.py \
    --mode=local \
    --config_bundle=omnisearchsage.configs.configs.OmniSearchSageTrainingConfigBundle
    """

    trainer_config: TrainerConfig = TrainerConfig(
        trainer_class="omnisearchsage.train.OmniSearchSageTrainer",
        nccl_timeout_mins=10 * 60,
        namespace="omnisearchsage_train",
        s3_save_dir='',
    )

    resource_config: ResourceConfig = DEFAULT_TRAIN_RESOURCE_CONFIG
    app_config: OmniSearchSageAppConfig = OmniSearchSageAppConfig()


@dataclass
class OmniSearchSageEvalConfigBundle(ConfigBundle):
    """
    ConfigBundle for omnisearchsage model eval

     python3.8 omnisearchsage/launcher/launcher.py \
        --mode=local \
        --config_bundle=omnisearchsage.configs.configs.OmniSearchSageTrainingConfigBundle \
        --trainer_config.trainer_class=omnisearchsage.trackers.OmniSearchSageExpHeadTracker
    """

    trainer_config: TrainerConfig = TrainerConfig(
        trainer_class="placeholder", nccl_timeout_mins=60, namespace="omnisearchsage_eval", s3_save_dir=''
    )
    resource_config: ResourceConfig = DEFAULT_EVAL_RESOURCE_CONFIG
    app_config: OmniSearchSageAppConfig = OmniSearchSageAppConfig()
