from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List

import os.path

from ml_common.boto_s3 import download_if_s3
from omnisearchsage.common.solver.basic_solver import BasicSolver
from omnisearchsage.common.solver.basic_solver import EvalFunc
from omnisearchsage.common.solver.basic_solver import EvalOnlySolver
from omnisearchsage.common.trainers.distributed_trainer import PytorchDistributedTrainer
from omnisearchsage.feature_consts import STRING_FEATURES
from omnisearchsage.feature_consts import TENSOR_FEATURES
from omnisearchsage.train import create_eval_function
from omnisearchsage.train import create_model
from omnisearchsage.train import get_tokenizers_from_model
from torch import nn

if TYPE_CHECKING:
    from omnisearchsage.common.solver.solver import Solver
    from omnisearchsage.common.types import EntityType
    from omnisearchsage.configs.configs import OmniSearchSageAppConfig
    from omnisearchsage.configs.configs import OmniSearchSageEvalConfigBundle
    from omnisearchsage.model import OmniSearchSAGE


class FakeDataParallel(nn.Module):
    """
    Wrapper to fake some of the attributes and behaviors of DataParallel. This
    is to simplify device agnostic code logic

    We need the following:

    Attributes:
        module - nn.Module that is wrapped by DataParallel

    Methods:
        forward - calls module(input)
    """

    def __init__(self, module):
        super(FakeDataParallel, self).__init__()
        self.module = module

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)


class OmniSearchSageHEADTrackerBase(PytorchDistributedTrainer):
    """
    Base class for OmniSearchSAGE HEAD tracker. subclasses should specify snapshot path and model.

    Assumes the presence of attributes text_embedder and aggregator on the value returned by get_model
    """

    TOKENIZE_IN_MODEL = True
    config_bundle: OmniSearchSageEvalConfigBundle

    @property
    def artifact_path(self) -> str:
        raise NotImplementedError

    def get_model(self) -> OmniSearchSAGE:
        """
        Get an instance of the model to evaluate
        """
        raise NotImplementedError

    @property
    def tensor_features(self) -> Dict[EntityType, List[str]]:
        return TENSOR_FEATURES

    @property
    def string_features(self) -> Dict[EntityType, List[str]]:
        return STRING_FEATURES

    @property
    def app_config(self) -> OmniSearchSageAppConfig:
        return self.config_bundle.app_config

    def create_eval_fn(self, model: OmniSearchSAGE):
        return create_eval_function(
            model=model,
            string_feature_names=self.string_features,
            num_workers=self.app_config.num_workers,
            tokenizers=get_tokenizers_from_model(model) if not self.TOKENIZE_IN_MODEL else {},
            subdims=(256,),
            index_batch_size=1024 * 8,
            pair_batch_size=1024 * 2,
            index_size=self.app_config.eval_index_size,
        )

    def create_solvers(self) -> Solver:
        assert self.world_size == 1
        # Load model
        model = self.get_model()

        eval_fn = self.create_eval_fn(model)

        return EvalOnlySolver(
            eval_fn=EvalFunc.from_eval_fn(eval_fn, iterations=0),
        )


class OmniSearchSageExpHeadTracker(OmniSearchSageHEADTrackerBase):
    """

     python3.8 omnisearchsage/launcher/launcher.py \
        --mode=local \
        --config_bundle=omnisearchsage.configs.configs.OmniSearchSageTrainingConfigBundle \
        --trainer_config.trainer_class=omnisearchsage.trackers.OmniSearchSageExpHeadTracker
    """

    TOKENIZE_IN_MODEL = False

    @property
    def artifact_path(self) -> str:
        return ""

    def get_model(self) -> OmniSearchSAGE:
        local_snapshot_path = download_if_s3(self.artifact_path, outdir=os.path.join(self.base_dir, "models"))
        model = create_model(self.app_config.query_base_model_name, self.device)
        BasicSolver.load_pretrain_weights(FakeDataParallel(model), local_snapshot_path)
        return model
