from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple

import logging
import os
import random

import numpy as np
import pyarrow
import torch
from apex.optimizers import FusedLAMB
from omnisearchsage.common.eval_utils.containers import EvaluationLoaderGroup
from omnisearchsage.common.solver.basic_solver import BasicSolver
from omnisearchsage.common.solver.basic_solver import EvalFunc
from omnisearchsage.common.trainers.distributed_trainer import PytorchDistributedTrainer
from omnisearchsage.common.types import EntityType
from omnisearchsage.common.utils.lr_scheduler import CosineAnnealingLRScheduler
from omnisearchsage.common.utils.lr_scheduler import GradualWarmupCompositeLRScheduler
from omnisearchsage.configs.configs import OmniSearchSageTrainingConfigBundle
from omnisearchsage.datasets import RandomDataset
from omnisearchsage.datasets import SageMultiIterator
from omnisearchsage.eval import create_evaluation
from omnisearchsage.feature_consts import ITEM_FEATURES_IMAGE_LEVEL
from omnisearchsage.feature_consts import STRING_FEATURES
from omnisearchsage.feature_consts import TENSOR_FEATURE_TO_EMB_SIZE
from omnisearchsage.feature_consts import TENSOR_FEATURES
from omnisearchsage.feature_consts import VISUAL_FEATURE_NAME
from omnisearchsage.model import ConcatInput
from omnisearchsage.model import ItemFeatureSummarizer
from omnisearchsage.model import OmniSearchSAGE
from omnisearchsage.model import OmniSearchSAGEPinEmbedder
from omnisearchsage.model import OmniSearchSAGEQueryEmbedder
from omnisearchsage.model import PinTextEmbedder
from omnisearchsage.model import TowerState
from omnisearchsage.model import UEFeatureDecoder
from omnisearchsage.modules.sage_embedder import FeatureEmbedder
from omnisearchsage.modules.text_embedding import HashEmbeddingBag
from omnisearchsage.modules.text_embedding import TextEmbedder
from omnisearchsage.modules.text_embedding import _generate_mlp
from omnisearchsage.modules.tokenization import BertTokenizerWrapper
from omnisearchsage.modules.tokenization import MultiVocabTokenizer
from omnisearchsage.modules.tokenization import TextNormalizationOption
from torch import nn
from torch.distributed import barrier
from torch.distributed import get_rank
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers import AutoConfig
from transformers import AutoModel

if TYPE_CHECKING:
    from omnisearchsage.common.solver.solver import Solver
    from omnisearchsage.configs.configs import OmniSearchSageAppConfig
    from torch.utils.tensorboard import SummaryWriter

logging.getLogger("botocore").setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)


def get_tensor_features() -> Dict[EntityType, List[str]]:
    return TENSOR_FEATURES


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


INDEX_SIZE_MAP = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000,
    "1M": 1_000_000,
    "10M": 10_000_000,
}


def create_eval_function(
    model: torch.nn.Module,
    tokenizers: Dict[EntityType, nn.Module],
    string_feature_names: Dict[EntityType, List[str]],
    num_workers: int,
    subdims: Sequence[int] = (16, 32, 64, 128, 256, 512),
    pair_batch_size: int = 128,
    index_batch_size: int = 4096,
    index_size: str = "10M",
) -> Callable[[int, str, Optional[SummaryWriter]], None]:
    # Batch size of 1 to split query and positive
    rank = get_rank() if torch.distributed.is_initialized() else 0

    def create_pair_loader(entity_type: EntityType) -> DataLoader:
        iterator = RandomDataset(
            batch_size=pair_batch_size,
            string_features=string_feature_names,
            query_vocab_size=len(tokenizers[EntityType.SEARCH_QUERY]),
            vocab_size=len(tokenizers[entity_type]),
            candidate_entity_type=entity_type,
            negative_ratio=0,
            num_examples=80,
        )
        return DataLoader(
            dataset=iterator,
            num_workers=num_workers,
            worker_init_fn=_get_worker_init_fn(num_workers, rank),
            batch_size=None,
        )

    def create_index_loader(entity_type: EntityType) -> DataLoader:
        iterator = RandomDataset(
            batch_size=index_batch_size if entity_type != EntityType.SEARCH_QUERY else 16,
            string_features=string_feature_names,
            query_vocab_size=len(tokenizers[EntityType.SEARCH_QUERY]),
            vocab_size=len(tokenizers[entity_type]),
            candidate_entity_type=entity_type,
            negative_ratio=0,
            num_examples=INDEX_SIZE_MAP[index_size],
        )
        return DataLoader(
            dataset=iterator,
            num_workers=num_workers,
            worker_init_fn=_get_worker_init_fn(num_workers, rank),
            batch_size=None,
        )

    eval_loaders = [
        EvaluationLoaderGroup(
            index_name="pins",
            query_positive_loaders={
                "save": create_pair_loader(EntityType.SIGNATURE),
                "long_click": create_pair_loader(EntityType.SIGNATURE),
                "relevance": create_pair_loader(EntityType.SIGNATURE),
            },
            index_loader=create_index_loader(EntityType.SIGNATURE),
        ),
        EvaluationLoaderGroup(
            index_name="items",
            query_positive_loaders={
                "save": create_pair_loader(EntityType.ITEM),
                "long_click": create_pair_loader(EntityType.ITEM),
                "offsite": create_pair_loader(EntityType.ITEM),
            },
            index_loader=create_index_loader(EntityType.ITEM),
        ),
        EvaluationLoaderGroup(
            index_name="queries",
            query_positive_loaders={
                "click": create_pair_loader(EntityType.SEARCH_QUERY),
            },
            index_loader=create_index_loader(EntityType.SEARCH_QUERY),
        ),
    ]

    return create_evaluation(
        model,
        eval_groups=eval_loaders,
        subdims=subdims,
    )


def get_mlp_dims(feature_names: str, text_embedding_dim: int = 256) -> Sequence[int]:
    """
    Function that returns a sequence for MLP layer in the model
    """

    return (
        sum(TENSOR_FEATURE_TO_EMB_SIZE[feature] for feature in feature_names) + text_embedding_dim,
        1024,
        1024,
        512,
    )


def union_features(list_of_features: List[Dict[EntityType, List[str]]]) -> Dict[EntityType, List[str]]:
    """
    Union a list of features
    """
    result = {}
    for features in list_of_features:
        for entity_type, feature_list in features.items():
            if entity_type not in result:
                result[entity_type] = []
            result[entity_type] += feature_list
    return {entity_type: list(set(feature_list)) for entity_type, feature_list in result.items()}


class DummyTokenizer(nn.Module):
    def __len__(self):
        return 1_000_000


def create_model(
    query_base_model_name: str,
    device: torch.device,
) -> OmniSearchSAGE:
    model_type = AutoConfig.from_pretrained(query_base_model_name).model_type
    if model_type in ["distilbert", "bert"]:
        text_tokenizer = BertTokenizerWrapper(
            query_base_model_name,
            max_sequence_length=64,
            text_normalization_options={
                TextNormalizationOption.TRIM_SPACE,
                TextNormalizationOption.COLLAPSE_WHITESPACE,
            },
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pin_text_tokenizer = MultiVocabTokenizer.default()
    vocab_size = len(pin_text_tokenizer)

    pin_text_embedder = HashEmbeddingBag(
        num_hashes=2,
        vocab_size=vocab_size,
        embedding_dim=256,
        num_embeds=100_000,
        hash_weights=False,
    )
    pin_item_encoder = _generate_mlp(
        *get_mlp_dims(feature_names=TENSOR_FEATURES[EntityType.SIGNATURE]),
        layernorm=True,
        normalize=True,
        precision=torch.float16,
    )

    def create_embedder(
        visual_feature_name: str,
        string_feature_names: List[str],
        tensor_feature_names: List[str],
        entity_encoder: nn.Module,
        features_to_summarize: List[str] = None,
    ):
        modules = [UEFeatureDecoder(visual_feature_name=visual_feature_name)]
        input_tensor_feature_names = tensor_feature_names
        if features_to_summarize:
            print(f"Summarizing features at input: {features_to_summarize}, {tensor_feature_names}")
            new_tensor_feats, new_features_to_summarize = [], []
            mask_features = set()
            for f in tensor_feature_names:
                feat_name = f[: -len("_feat")]
                if f.endswith("_feat") and feat_name in features_to_summarize:
                    new_tensor_feats.append(feat_name)
                    new_features_to_summarize.append(feat_name)
                    mask_features.add(feat_name + "_mask")
                else:
                    new_tensor_feats.append(f)
            new_tensor_feats = [f for f in new_tensor_feats if f not in mask_features]
            tensor_feature_names, features_to_summarize = new_tensor_feats, new_features_to_summarize
            print(f"Summarizing features at output: {features_to_summarize}, {tensor_feature_names}")

            modules.append(ItemFeatureSummarizer(features_to_summarize=features_to_summarize))
        modules.extend(
            [
                PinTextEmbedder(
                    embedding_bag=pin_text_embedder,
                    feature_names=string_feature_names,
                    output_feature_name="_text_embedding",
                ),
                ConcatInput(feature_names=["_text_embedding"] + tensor_feature_names),
                entity_encoder,
            ]
        )
        return OmniSearchSAGEPinEmbedder(
            embedder=nn.Sequential(*modules),
            string_feature_names=string_feature_names,
            tensor_feature_names=input_tensor_feature_names,
            tensor_feature_to_emb_size=TENSOR_FEATURE_TO_EMB_SIZE,
            tokenizer=pin_text_tokenizer,
            device=device,
        )

    pin_embedder = create_embedder(
        visual_feature_name=VISUAL_FEATURE_NAME,
        string_feature_names=STRING_FEATURES[EntityType.SIGNATURE],
        tensor_feature_names=TENSOR_FEATURES[EntityType.SIGNATURE],
        entity_encoder=pin_item_encoder,
    )

    item_embedder = create_embedder(
        visual_feature_name=VISUAL_FEATURE_NAME + "_feat",
        string_feature_names=STRING_FEATURES[EntityType.ITEM],
        tensor_feature_names=TENSOR_FEATURES[EntityType.ITEM],
        features_to_summarize=ITEM_FEATURES_IMAGE_LEVEL,
        entity_encoder=pin_item_encoder,
    )

    query_embedder = OmniSearchSAGEQueryEmbedder(
        embedder=TextEmbedder(
            query_base_model_name,
            vocab_size=len(text_tokenizer),
            pooling_mode="cls" if model_type not in ["t5", "mt5", "umt5"] else "mean",
            input_id_feat_name="query_text_input_ids",
            attention_mask_feat_name="query_text_attention_mask",
            precision=torch.float16,
            output_dim=512,
        ),
        tokenizer=text_tokenizer,
        feature_name="query_text",
        device=device,
    )

    embedders = {
        (EntityType.SEARCH_QUERY, TowerState.LEARNED): query_embedder,
        (EntityType.SIGNATURE, TowerState.LEARNED): pin_embedder,
        (EntityType.SIGNATURE, TowerState.FIXED_GS): FeatureEmbedder("gs_v5", precision=torch.float16),
        (EntityType.ITEM, TowerState.FIXED_IS): FeatureEmbedder("item_is_v2", precision=torch.float16),
        (EntityType.ITEM, TowerState.LEARNED): item_embedder,
    }

    model = OmniSearchSAGE(embedders=embedders, device=device)
    return model


def get_tokenizers_from_model(model: OmniSearchSAGE) -> Dict[EntityType, nn.Module]:
    tokenizers = {}
    for k, v in model.embedders.items():
        entity_type, tower_state = model.decode_embedder_key(k)
        if tower_state != TowerState.LEARNED:
            continue
        assert entity_type not in tokenizers, f"Duplicate tokenizer for {EntityType._VALUES_TO_NAMES[entity_type]}"
        tokenizers[entity_type] = v.tokenizer
    return tokenizers


class OmniSearchSageTrainer(PytorchDistributedTrainer):
    """
    SearchSage Trainer

    >> ./scripts/bin/idoc "python omnisearchsage/launcher.py \
        --mode=local \
        --resource_config.gpus_per_node=2 \
        --app_config.num_workers=3 \
        --app_config.batch_size=1024 \
        --config_bundle=omnisearchsage.configs.configs.SearchSageTrainingConfigBundle"

        ./scripts/bin/idoc "python trainer/ppytorch/utils/launcher.py \
        --mode=tcp \
        --tcp_config.job_name=ss-sample-tcp \
        --config_bundle=ml_resources.mlenv.searchsage.configs.SearchSageTrainingConfigBundle"

        ./scripts/bin/idoc "python trainer/ppytorch/utils/launcher.py \
        --mode=local \
        --config_bundle=ml_resources.mlenv.searchsage.configs.SearchSageTrainingConfigBundle"

    """

    config_bundle = OmniSearchSageTrainingConfigBundle

    @property
    def app_config(self) -> OmniSearchSageAppConfig:
        return self.config_bundle.app_config

    def setup_model_and_optimizer(self) -> Tuple[OmniSearchSAGE, torch.optim.Optimizer]:
        model = create_model(
            query_base_model_name=self.app_config.query_base_model_name,
            device=self.device,
        )
        # Load optimizer
        model_params = list(model.parameters())
        optimizer = FusedLAMB(model_params, lr=self.base_lr)

        return model, optimizer

    def create_data_loader(self, tokenizers) -> DataLoader:
        string_feature_names = STRING_FEATURES
        query_vocab_size = len(tokenizers[EntityType.SEARCH_QUERY])
        vocab_size = len(tokenizers[EntityType.SIGNATURE])

        task_weights = {
            "pin_onsite": (EntityType.SIGNATURE, 8),
            "item_onsite": (EntityType.ITEM, 4),
            "query_click": (EntityType.SEARCH_QUERY, 2),
            "item_offsite": (EntityType.SIGNATURE, 2),
        }
        total_weight = sum(w for _, (entity_type, w) in task_weights.items())
        for task_name, (entity_type, w) in task_weights.items():
            task_weights[task_name] = (entity_type, int(w * self.app_config.batch_size / total_weight))

        dataset_iterators = [
            RandomDataset(
                batch_size=batch_size,
                string_features=string_feature_names,
                query_vocab_size=query_vocab_size,
                negative_ratio=self.app_config.neg_ratio,
                vocab_size=vocab_size,
                candidate_entity_type=candidate_entity_type,
            )
            for task_name, (candidate_entity_type, batch_size) in task_weights.items()
        ]

        return DataLoader(
            SageMultiIterator(dataset_iterators),
            num_workers=self.num_workers,
            worker_init_fn=_get_worker_init_fn(self.num_workers, self.rank),
            pin_memory=True,
            batch_size=None,
        )

    def _download_query_base_models(self) -> None:
        if get_rank() != 0:
            return
        BertTokenizerWrapper(
            self.app_config.query_base_model_name,
            max_sequence_length=64,
            text_normalization_options={
                TextNormalizationOption.UNICODE_NORMALIZE,
                TextNormalizationOption.TRIM_SPACE,
                TextNormalizationOption.COLLAPSE_WHITESPACE,
            },
        )
        AutoModel.from_pretrained(self.app_config.query_base_model_name)

    def create_solver(self) -> Solver:
        pyarrow.set_cpu_count(1)

        # make sure we don't get have issues with all processes trying to download the same files
        self._download_query_base_models()
        barrier()

        model, optimizer = self.setup_model_and_optimizer()
        tokenizers = get_tokenizers_from_model(model)

        eval_fn = create_eval_function(
            model=model,
            string_feature_names=STRING_FEATURES,
            tokenizers=tokenizers,
            subdims=(256,),
            pair_batch_size=self.app_config.batch_size,
            index_batch_size=self.app_config.batch_size * 4,
            index_size=self.app_config.eval_index_size,
            num_workers=self.num_workers,
        )

        def after_scheduler_fn(opt) -> CosineAnnealingLRScheduler:
            return CosineAnnealingLRScheduler(opt, self.iterations)

        lr_scheduler = GradualWarmupCompositeLRScheduler(
            optimizer,
            num_warmup_steps=self.app_config.warmup_steps,
            after_scheduler_fn=after_scheduler_fn,
        )

        if self.rank == 0:
            print(model)

        return BasicSolver(
            model=DistributedDataParallel(model),
            optimizer=optimizer,
            train_dataset_loader=self.create_data_loader(tokenizers),
            iterations=self.iterations,
            batch_size=self.app_config.batch_size,
            snapshot_every_n_iter=self.app_config.eval_every_n_iter,
            eval_at_start=True,
            eval_func=EvalFunc.from_eval_fn(eval_fn, iterations=self.iterations),
            eval_every_n_iter=self.app_config.eval_every_n_iter,
            lr_scheduler=lr_scheduler,
            precision=torch.bfloat16,
            summarize_func=lambda x: x,
            summarize_every_n_iter=50,
            max_grad_norm=self.app_config.max_grad_norm,
            model_forward_func=lambda m, batch: m(batch),
        )

    @property
    def iterations(self) -> int:
        return self.app_config.iterations

    @property
    def num_workers(self) -> int:
        return self.app_config.num_workers

    @property
    def base_lr(self) -> float:
        return 10**self.app_config.log10_base_lr
