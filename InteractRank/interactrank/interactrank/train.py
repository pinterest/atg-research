from __future__ import annotations

from typing import Callable
from typing import Dict
from typing import List

import functools
import logging
import os
from functools import partial

import numpy as np
import random
import pyarrow as pa
import torch
from apex.optimizers import FusedAdam
from torch.nn.parallel import DistributedDataParallel
from interactrank.common.trainers.distributed_trainer import PytorchDistributedTrainer
from interactrank.common.solver.solver import is_root_process
from interactrank.common.layers import generate_embedding_vocabs
from interactrank.common.eval.basic_evaluator import BasicEvaluator
from interactrank.common.solver.basic_solver import BasicSolver
from interactrank.common.solver.basic_solver import EvalFunc
from interactrank.common.utils.utils import send_to_device
from interactrank.eval import get_accumulators
from interactrank.eval import two_tower_lightweight_post_inference_fn
from interactrank.eval import (
    two_tower_tabular_lightweight_inference_fn,
)
from interactrank.label_extractor import TabularLightweightLabelExtractor
from interactrank.model import CrossLayer
from interactrank.model import SearchTwoTowerDeployableModel
from interactrank.model import Tower
from interactrank.data.lw_features import Feature, FeatureDefinition
from interactrank.data.stats_metadata import FeatureStatsMetadata
from interactrank.common.utils.utils import drop_negative_ids
from interactrank.common.lr_scheduler import get_constant_schedule
from interactrank.common.solver.solver import Solver
from interactrank.configs.bundle import SearchMultiPassAppLwConfig
from interactrank.feature_consts import FEATURE_NAME_TO_TYPE_MAPPING
from interactrank.constants.base_constants import DEFAULT_EMBEDDING_DIM
from interactrank.constants.base_constants import DEFAULT_HIDDEN_SIZES
from interactrank.constants.base_constants import DEFAULT_NUM_EVAL_WORKER
from interactrank.constants.base_constants  import DEFAULT_NUM_WORKER
from interactrank.constants.base_constants  import DEFAULT_SNAPSHOT_N_ITER
from interactrank.constants.base_constants import DEFAULT_TIMESTAMP_FEAT_NAME
from interactrank.configs.base_configs import SEARCH_LW_HEAD_CONFIGS
from interactrank.constants.base_constants import LABEL_COLUMNS
from interactrank.configs.bundle import SearchLwEngagementTabularTrainerConfigBundle
from interactrank.common.eval.distributed_processor import all_gather_object
from interactrank.data.dataset import RandomDataset
from torch.utils.data import DataLoader
from interactrank.data.dataset import LwMultiIterator
from interactrank.common.types import EntityType
from interactrank.feature_consts import TENSOR_FEATURES


NUM_FILES_FOR_ESTIMATION_FEATURE_STATS = 100
NUM_FILES_FOR_ESTIMATION_FEATURE_MONITOR = 10000
SHARD_COL = "user_id"

logging.getLogger("botocore").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)



class EmbeddingBagDict(torch.nn.Module):
    def __init__(self, emb_name_to_vocab):
        super().__init__()
        self.embeddings = torch.nn.ModuleDict({
            name: torch.nn.EmbeddingBag(
                num_embeddings=len(vocab),
                embedding_dim=DEFAULT_EMBEDDING_DIM,
                mode="mean"
            ) for name, vocab in emb_name_to_vocab.items()
        })

    def forward(self, formatted_data):
        embedded_features = {}
        for name, embedding in self.embeddings.items():
            if name in formatted_data:
                embedded_features[name] = embedding(formatted_data[name])
        # Add features not found in emb_name_to_vocab as-is
        for name, feature in formatted_data.items():
            if name not in self.embeddings:
                embedded_features[name] = feature
        return embedded_features

def _get_worker_init_fn(num_workers: int, rank: int) -> Callable[[int], None]:
    def worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        pid = os.getpid()
        # time.sleep(random.random() + worker_id )
        logger.info(f"Initializing worker: {worker_id + 1}/{num_workers} for rank {rank}, pid={pid}.")
        logger.info(f"Successfully initialized worker: {worker_id + 1}/{num_workers} for rank {rank}, pid={pid}.")

    return worker_init_fn

class SearchLwsTrainer(PytorchDistributedTrainer):
    config_bundle = SearchLwEngagementTabularTrainerConfigBundle()
    root_run_dir = None
    @property
    def app_config(self) -> SearchMultiPassAppLwConfig:
        return self.config_bundle.app_config

    def get_random_dataset(self, batch_size:int = 1024) -> RandomDataset:
        return RandomDataset(
                batch_size=batch_size,
                negative_ratio=self.app_config.neg_ratio,
            )

    def create_data_loader(self, random_dataset: RandomDataset) -> DataLoader:
        dataset_iterators = random_dataset

        return DataLoader(
            LwMultiIterator(dataset_iterators),
            num_workers=self.num_workers,
            worker_init_fn=_get_worker_init_fn(self.num_workers, self.rank),
            pin_memory=True,
            batch_size=None,
        )

    def get_root_run_dir(self, run_dir: str) -> str:
        """
        Gets the run directory of the root (rank 0) process and caches it in self.root_run_dir.  Each process
        passes in its own run directory, the return value is the root run directory.
        Args:
            run_dir: the run directory of the current process (any rank)
        Returns: The run directory of the root process.
        """
        if self.root_run_dir is None:
            run_dirs = all_gather_object(run_dir)
            self.root_run_dir = run_dirs[0]
        return self.root_run_dir

    def create_evaluator(
        self,
        task: str,
        is_cpu_inference: bool = False,
        is_training_eval_loop: bool = True,
        all_heads: List = None,
    ) -> EvalFunc:
        random_dataset_eval = self.get_random_dataset(500)
        self.eval_dataloader_iter = self.create_data_loader(random_dataset_eval).__iter__()
        next(self.eval_dataloader_iter)

        eval_inference_fn = partial(
            two_tower_tabular_lightweight_inference_fn,
            metamodel_weights=list(map(float, self.app_config.metamodel_weights.split(","))),
            saved_results_cols=[],
            task=task,
        )

        def tabular_engagement_evaluator_fn(iteration: int, **kwargs) -> BasicEvaluator:
            # If the model is intended to be used in GPU inference in serving, evaluate on GPU. Otherwise,
            # eval will run on CPU using tabular_eval_cpu_inference_fn.
            max_num_eval_iters = self.app_config.max_num_eval_iterations
            is_final_evaluation = iteration >= self.app_config.iterations and is_training_eval_loop
            if is_final_evaluation:
                # Run for a different number of iterations for final evaluation
                max_num_eval_iters = self.app_config.max_num_final_eval_iterations
            inference_fn = (
                eval_inference_fn
                if not is_cpu_inference
                else (lambda batch, _: send_to_device(batch, torch.device("cuda"), ignore_error=True))
            )
            return BasicEvaluator(
                model=self.model.module,
                # As we stream infinitely, just pass in the DataLoader iterator to BasicEvaluator to avoid
                # calls that repeatedly construct the iterator (and call S3). This suffices for BasicEvaluator's
                # interfaces.
                dataloader=self.eval_dataloader_iter, #if not is_final_evaluation else self.final_eval_dataloader_iter,
                inference_fn=inference_fn,
                accumulators=get_accumulators(
                    all_heads,
                    self.app_config.enable_cross_features,
                    self.get_root_run_dir(kwargs["run_dir"]),
                ),
                post_inference_fn=functools.partial(
                    two_tower_lightweight_post_inference_fn,
                    iteration=iteration,
                    run_dir=kwargs["run_dir"],
                    metric_prefix="engagement",
                    world_size=self.world_size,
                    device=self.device,
                    model=self.model.module,
                    enable_compute_recall=False,
                    enable_estimator_rewards=self.app_config.enable_estimator_rewards,
                ),
                max_num_eval_iters=max_num_eval_iters,
            )
        output_eval_function = {"engagement": tabular_engagement_evaluator_fn}
        # # return eval if flag is set to True
        return EvalFunc.from_evaluator_dict(output_eval_function) if self.app_config.should_eval else None

    @staticmethod
    def get_embedding_layer(feature_map, shared_embedding_voc_min_count):
        """
        :param feature_map: feature map used in the model
        :param shared_embedding_voc_min_count: shared embedding vocab min count
        :return: embedding layer used in tower definition & emb_name_to_vocab
        """
        if shared_embedding_voc_min_count is None:
            shared_embedding_voc_min_count = {"default": 150}
        feat_to_emb_name, emb_name_to_vocab = generate_embedding_vocabs(feature_map, shared_embedding_voc_min_count)
        # drop negative ids
        emb_name_to_vocab = drop_negative_ids(emb_name_to_vocab)
        embedding_layer = EmbeddingBagDict(emb_name_to_vocab)
        return embedding_layer, emb_name_to_vocab

    def get_feature_maps_and_feature_stats(self, random_dataset: RandomDataset) -> Dict[str, Feature]:
        feature_map = {}
        for entity_type in EntityType:
            if entity_type in TENSOR_FEATURES:
                for feature_name in TENSOR_FEATURES[entity_type]:
                    new_feature_name = str(entity_type.value)+"/"+feature_name
                    feature_map[new_feature_name] = Feature()
                    feature_map[new_feature_name].definition = FeatureDefinition()
                    feature_map[new_feature_name].definition.feature_type = FEATURE_NAME_TO_TYPE_MAPPING.get(feature_name)
                    feature_map[new_feature_name].definition.entity_type = entity_type
                    feature_map[new_feature_name].definition.name = feature_name
                    feature_map[new_feature_name].metadata = FeatureStatsMetadata()
                    # Get feature tensor from the batch and compute stats
                    feature_tensor = random_dataset._batch.batch[new_feature_name]
                    feature_map[new_feature_name].definition.shape = feature_tensor.shape
                    feature_map[new_feature_name].metadata.min_value = torch.min(feature_tensor)
                    feature_map[new_feature_name].metadata.max_value = torch.max(feature_tensor)
                    feature_map[new_feature_name].metadata.std = torch.std(feature_tensor.to(torch.float32))
                    feature_map[new_feature_name].metadata.mean = torch.mean(feature_tensor.to(torch.float32))
                    unique_values, counts = torch.unique(feature_tensor, return_counts=True)
                    feature_map[new_feature_name].metadata.vocab = dict(zip(unique_values.tolist(), counts.tolist()))
                    if entity_type == EntityType.SIGNATURE:
                        feature_map[new_feature_name].metadata.signal_group = "item_group"
                    elif entity_type == EntityType.CROSS:
                        feature_map[new_feature_name].metadata.signal_group = "item_query_group"
                    elif entity_type == EntityType.SEARCH_QUERY:
                        feature_map[new_feature_name].metadata.signal_group = "query_group"
                    if feature_name not in ("category_count", "sig_count"):
                        feature_map[new_feature_name].metadata.signal_type = feature_name + "_"

        return feature_map

    def get_feature_map_for_tower(self, feature_map: Dict[str, Feature], tower_feature_list: List[str], entity_type):
        new_feature_map = {}
        for feature_name in tower_feature_list:
            new_feature_map[str(entity_type.value)+ "/" + feature_name] = feature_map[str(entity_type.value) + "/" + feature_name]
        return new_feature_map

    def create_solver(self) -> Solver:
        """
        Setup the solver for the training task.
        :return: The solver for the particular worker.
        """

        self.label_set = LABEL_COLUMNS
        if self.gpus and torch.cuda.is_available():
            # Enable auto cudnn tuner for fixed input size
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(self.gpus[0])
            device = torch.device("cuda")
            print("gpu")

        self.label_extractor = TabularLightweightLabelExtractor(
            device=device,
            app_config=self.app_config,
        )
        self.engagement_heads = self.label_extractor.engagement_heads
        random_dataset = self.get_random_dataset(50000)
        feature_map = self.get_feature_maps_and_feature_stats(random_dataset)
        embedding_layer, emb_name_to_vocab = self.get_embedding_layer(feature_map, None)
        # ger respective tower feature maps
        context_tower_feature_map = self.get_feature_map_for_tower(feature_map, TENSOR_FEATURES[EntityType.SEARCH_QUERY], EntityType.SEARCH_QUERY)
        candidate_tower_feature_map = self.get_feature_map_for_tower(feature_map, TENSOR_FEATURES[EntityType.SIGNATURE], EntityType.SIGNATURE)
        cross_tower_feature_map = self.get_feature_map_for_tower(feature_map, TENSOR_FEATURES[EntityType.CROSS], EntityType.CROSS)
        # Define Query tower
        context_tower = Tower(
            features=TENSOR_FEATURES[EntityType.SEARCH_QUERY],
            feature_map = context_tower_feature_map,
            hidden_sizes=DEFAULT_HIDDEN_SIZES,
            embedding_layer=embedding_layer,
            emb_name_to_vocab=emb_name_to_vocab,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            time_feature=DEFAULT_TIMESTAMP_FEAT_NAME,
            enable_user_sequence=self.app_config.enable_user_sequence,
            tower_type="query",
            use_skip_connections=self.app_config.use_skip_connections,
            enable_pmn=self.app_config.enable_pmn,
            num_heads=len(SEARCH_LW_HEAD_CONFIGS)
            )
        # Define Pin Tower
        candidate_tower = Tower(
            features=TENSOR_FEATURES[EntityType.SIGNATURE],
            feature_map=candidate_tower_feature_map,
            embedding_layer=embedding_layer,
            hidden_sizes=DEFAULT_HIDDEN_SIZES,
            emb_name_to_vocab=emb_name_to_vocab,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            time_feature=DEFAULT_TIMESTAMP_FEAT_NAME,
            enable_user_sequence=False,
            tower_type="item",
            use_skip_connections=self.app_config.use_skip_connections,
            enable_pmn=self.app_config.enable_pmn,
            num_heads=1,
        )

        cross_layer = CrossLayer(
                features=TENSOR_FEATURES[EntityType.CROSS],
                feature_map= cross_tower_feature_map,
                enable_compute_average_navboost=self.app_config.enable_compute_average_navboost,
        )

        self.model = SearchTwoTowerDeployableModel(
            label_map=self.label_set,
            label_extractor=self.label_extractor,
            device=device,
            context_tower=context_tower,
            candidate_tower=candidate_tower,
            cross_layer=cross_layer,
            use_in_batch_negatives=self.app_config.use_in_batch_negatives,
            in_batch_neg_rel_weight=self.app_config.in_batch_negative_weight,
            in_batch_neg_mask_on_queries=self.app_config.in_batch_neg_mask_on_queries,
            correct_sample_probability=self.app_config.correct_sample_probability,
            searchsage_eval_only_with_version=None,
            logged_relevance_loss_weight=self.app_config.logged_relevance_loss_weight,
            loss_function="focal" if self.app_config.use_focal_loss else "bce",
            upweight_tprc_factor=self.app_config.upweight_tprc_factor,
            batch_size=self.app_config.batch_size,
        )
        train_dataset_loader = self.create_data_loader(random_dataset)
        self.model.init(next(iter(train_dataset_loader)))
        if is_root_process():
            print(self.model)

        for k, v in self.model.state_dict().items():
            if torch.nn.parameter.is_lazy(v):
                logger.info(f"Param {k} is not initialized !")

        # Create evaluation function with non-distributed model
        eval_func = self.create_evaluator(
            self.app_config.task,
            is_cpu_inference=self.device.type == "cpu",
            is_training_eval_loop=True,
            all_heads=self.engagement_heads,
        )
        # setup model for distributed training
        self.model = DistributedDataParallel(self.model, find_unused_parameters=True)
        optimizer = FusedAdam(self.model.parameters(), lr=self.app_config.lr, betas=(0.9, 0.999))
        lr_scheduler = get_constant_schedule(optimizer)
        # Create Solver
        return BasicSolver(
            model=self.model,
            optimizer=optimizer,
            batch_size=self.app_config.batch_size,
            max_grad_norm=5.0,
            snapshot_every_n_iter=DEFAULT_SNAPSHOT_N_ITER,
            snapshot_filename_prefix="snap",
            model_forward_func=lambda m, b: m(b),
            train_dataset_loader=train_dataset_loader,
            lr_scheduler=lr_scheduler,
            eval_func=eval_func,
            eval_every_n_iter=self.app_config.eval_every_n_iter,
            eval_at_end=self.app_config.should_eval,
            iterations=self.app_config.iterations,
            summarize_every_n_iter=self.app_config.summarize_every_n_iter,
            summarize_func=lambda x: x,
        )
    @property
    def num_workers(self) -> int:
        return DEFAULT_NUM_WORKER

    @property
    def num_workers_eval(self) -> int:
        return DEFAULT_NUM_EVAL_WORKER
