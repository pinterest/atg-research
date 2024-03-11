from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from collections import defaultdict
from enum import IntEnum

import simplejson as json
import torch
from omnisearchsage.common.solver.basic_solver import BasicSolver
from omnisearchsage.common.types import EntityType
from omnisearchsage.modules.metric_learning_softmax import MultiDimSoftmaxCorrectionLossV2
from omnisearchsage.modules.sage_embedder import VisualFeatureEmbedder
from torch import nn

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.tensorboard import SummaryWriter
    from trainer.ppytorch.data.sage_features import SageBatch


def tensor_feats_to(
    tensor_feats: Dict[EntityType, Dict[str, torch.Tensor]],
    *args,
    **kwargs,
) -> Dict[EntityType, Dict[str, torch.Tensor]]:
    if tensor_feats is None:
        return None

    out = {}
    for entity_type, entity_feats in tensor_feats.items():
        out[entity_type] = {}
        for k, v in entity_feats.items():
            out[entity_type][k] = v.to(*args, **kwargs, non_blocking=True)

    return out


class OmniSearchSAGETrainTask(IntEnum):
    ORGANIC = 1
    ITEM = 2
    NATIVE = 3
    QUERY = 4
    ADS = 5
    SHOPPING_ADS = 6
    ADS_3P = 7


class TowerState(IntEnum):
    LEARNED = 0
    FIXED_GS = 1
    FIXED_MS = 2
    FIXED_IS = 3


class EmbedderWithTokenizer(nn.Module):
    def __init__(self, embedder: nn.Module, tokenizer: nn.Module, device: torch.device):
        super().__init__()
        self.embedder = embedder
        self.tokenizer = tokenizer
        self.should_tokenize = False
        self.device = device

    def encode(self, str_inputs: Dict[str, List[str]]) -> Dict[str, Tensor]:
        raise NotImplementedError

    def toggle_tokenization(self, tokenize: bool) -> EmbedderWithTokenizer:
        self.should_tokenize = tokenize
        return self

    def extra_repr(self):
        return f"should_tokenize={self.should_tokenize};device={self.device};"

    def forward(
        self, tensor_inputs: Dict[str, Tensor], str_inputs: Optional[Dict[str, List[str]]] = None
    ) -> torch.Tensor:
        if self.should_tokenize:
            assert str_inputs is not None
            text_ids = self.encode(str_inputs)
            tensor_inputs.update(text_ids)
        return self.embedder({k: v.to(self.device) for k, v in tensor_inputs.items()})


class OmniSearchSAGEQueryEmbedder(EmbedderWithTokenizer):
    def __init__(self, feature_name: str = "text", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_name = feature_name

    def extra_repr(self):
        return f"feature_name={self.feature_name}"

    def encode(self, str_inputs: Dict[str, List[str]]) -> Dict[str, Tensor]:
        tokenized_feat = self.tokenizer(str_inputs[self.feature_name])
        return {f"{self.feature_name}_{k}": v for k, v in tokenized_feat.items()}


class OmniSearchSAGEPinEmbedder(EmbedderWithTokenizer):
    def __init__(
        self,
        string_feature_names: List[str],
        tensor_feature_names: List[str],
        tensor_feature_to_emb_size: Dict[str, int],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.string_feature_names = string_feature_names
        self.tensor_feature_names = tensor_feature_names

    def extra_repr(self) -> str:
        return json.dumps(
            {
                "string_feature_names": self.string_feature_names,
                "tensor_feature_names": self.tensor_feature_names,
            },
            indent=4,
        )

    def encode(self, str_inputs: Dict[str, List[str]]) -> Dict[str, Tensor]:
        output = {}
        for feature_name in self.string_feature_names:
            tokenized_feat = self.tokenizer.encode_batch(str_inputs[feature_name])
            output.update({f"{feature_name}_{k}": v for k, v in tokenized_feat.items()})
        return output


class UEFeatureDecoder(nn.Module):
    def __init__(self, visual_feature_name: str):
        super().__init__()
        self.model = VisualFeatureEmbedder(key=visual_feature_name, use_unused_parameter=False)

    def extra_repr(self) -> str:
        return f"feature_name={self.model.key}"

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.model.key in feats, f"{self.model.key} not in feats ({feats.keys()})"
        feats[self.model.key] = self.model(feats)
        return feats


class ItemFeatureSummarizer(nn.Module):
    def __init__(self, features_to_summarize: List[str]):
        super().__init__()
        self.feature_to_summarize = features_to_summarize

    def extra_repr(self) -> str:
        return f"feature_to_summarize={self.feature_to_summarize}"

    # this is for item level features where the features are coming as (batch_size, x, dim)
    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for feat in self.feature_to_summarize:
            mask = feats.pop(feat + "_mask")
            feature_val = feats.pop(feat + "_feat")
            assert len(feature_val.shape) == 3, (feat, feature_val.shape)
            seq_len = mask.squeeze(dim=2).sum(dim=1, keepdim=True)
            seq_len[seq_len == 0] = 1
            feats[feat] = feature_val.sum(dim=1) / seq_len

        return feats


class PinTextEmbedder(nn.Module):
    def __init__(self, embedding_bag: nn.Module, feature_names: List[str], output_feature_name: str) -> None:
        super().__init__()
        self.embedding_bag = embedding_bag
        self.feature_names = feature_names
        self.output_feature_name = output_feature_name

    def extra_repr(self) -> str:
        return f"feature_names={self.feature_names}, output_feature_name={self.output_feature_name}"

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        embeds = []
        assert self.output_feature_name not in feats, f"{self.output_feature_name} already exists in feats"
        for name in self.feature_names:
            embeds.append(self.embedding_bag(feats[f"{name}_input_ids"], feats[f"{name}_offsets"]))
        feats[self.output_feature_name] = torch.sum(torch.stack(embeds), dim=0)
        return feats


class ConcatInput(nn.Module):
    def __init__(self, feature_names: List[str]):
        super().__init__()
        self.feature_names = feature_names

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([feats[feature] for feature in self.feature_names], dim=1)

    def extra_repr(self) -> str:
        return f"feature_names={self.feature_names}"


class OmniSearchSAGE(nn.Module):
    def __init__(self, embedders: Dict[Tuple[EntityType, TowerState], nn.Module], device: torch.device):
        super(OmniSearchSAGE, self).__init__()
        if device.type == "cuda":
            # Enable auto cudnn tuner for fixed input size
            torch.backends.cudnn.benchmark = True
        self.device = device

        self.entity_type_to_tower_states = {}
        for entity_type, tower_state in embedders:
            if entity_type in self.entity_type_to_tower_states:
                self.entity_type_to_tower_states[entity_type].append(tower_state)
            else:
                self.entity_type_to_tower_states[entity_type] = [tower_state]

        self.embedders = nn.ModuleDict({self.encode_embedder_key(k[0], k[1]): v for k, v in embedders.items()})
        self.similarity_loss = MultiDimSoftmaxCorrectionLossV2(
            synchronize_pos=True,
            w=2**26,
            d=2,
            # TODO: we need to revisit this given we have all tasks wont have same sum
            emb_dim_weights={k: v for k, v in [(x, 0.05) for x in [16, 32, 64, 128]] + [(256, 1.0), (512, 1.0)]},
            return_dict=True,
        )

        self.to(self.device)

    @staticmethod
    def encode_embedder_key(entity_type: EntityType, tower_state: TowerState) -> str:
        return f"{EntityType._VALUES_TO_NAMES[entity_type]}::{tower_state.name}"

    @staticmethod
    def decode_embedder_key(key: str) -> Tuple[EntityType, TowerState]:
        parts = key.split("::")
        return EntityType._NAMES_TO_VALUES[parts[0]], TowerState[parts[1]]

    @torch.no_grad()
    def _preprocess_input(self, batch: SageBatch):
        """
        Sends all the tensors in feats.query_feats and feats.pin_feats to self.device
        """
        return batch.replace(tensor_feats=tensor_feats_to(batch.tensor_feats, self.device))

    def _forward(self, batch: SageBatch, entity_type: EntityType, tower_state: TowerState):
        """
        Extract embeddings from feats for entity_type, assuming all tensors are on the correct device.
        """
        feats = batch.tensor_feats[entity_type]
        return self.embedders[self.encode_embedder_key(entity_type, tower_state)](feats)

    def compute_embeddings(
        self, batch: SageBatch, preprocess: bool = True
    ) -> Dict[(EntityType, TowerState), torch.Tensor]:
        embeddings = {}
        batch = self._preprocess_input(batch) if preprocess else batch
        for entity_type in batch.entity_types:
            # Skip entity types which are not processed by this model
            # For example, the v3alpha does not work with items where v3beta does
            if entity_type not in self.entity_type_to_tower_states:
                continue
            for tower_state in self.entity_type_to_tower_states[entity_type]:
                embeddings[entity_type, tower_state] = self._forward(batch, entity_type, tower_state)
        return embeddings

    def extract_query_embs(
        self, feats: SageBatch, embeddings: Dict[(EntityType, TowerState), torch.Tensor]
    ) -> Dict[str, Union[torch.Tensor, int]]:
        batch_size = feats.num_pairs
        query_emb = embeddings.pop((EntityType.SEARCH_QUERY, TowerState.LEARNED))
        assert query_emb.shape[0] >= batch_size, (query_emb.shape[0], batch_size)
        cand_start_index = 0
        if feats.candidate_entity_type == EntityType.SEARCH_QUERY:
            cand_start_index = batch_size
            embeddings[(EntityType.SEARCH_QUERY, TowerState.LEARNED)] = query_emb[batch_size:]
            query_emb = query_emb[:batch_size]

        return {
            "query_emb": query_emb,
            "cand_start_index": cand_start_index,
            "batch_size": batch_size,
        }

    def forward(self, all_feats: List[SageBatch]) -> Dict[str, Union[torch.Tensor, Dict[str, Tensor]]]:
        assert self.training, "Only call for training mode"
        all_query_features = defaultdict(list)
        all_pos_features = defaultdict(list)
        all_neg_features = defaultdict(list)
        all_pos_ids = defaultdict(list)
        all_neg_ids = defaultdict(list)
        all_query_ids = defaultdict(list)

        for feats in all_feats:
            # this make sure we dont compute any loss for completely filtered out batches
            if not feats.entity_types:
                continue
            feats = self._preprocess_input(feats)
            embeddings = self.compute_embeddings(feats, preprocess=False)
            assert feats.query_entity_type == EntityType.SEARCH_QUERY, feats.query_entity_type
            query_features = self.extract_query_embs(feats=feats, embeddings=embeddings)
            batch_size = query_features["batch_size"]
            query_ids = feats.search_query_feats["id_hash"][:batch_size]

            for (entity_type, tower_state), candidate_embs in embeddings.items():
                key = (feats.query_entity_type, entity_type, tower_state)
                cand_ids = feats.tensor_feats[entity_type]["id_hash"][query_features["cand_start_index"] :]
                all_pos_features[key].append(candidate_embs[:batch_size])
                all_neg_features[key].append(candidate_embs[batch_size:])
                all_pos_ids[key].append(cand_ids[:batch_size])
                all_neg_ids[key].append(cand_ids[batch_size:])
                all_query_features[key].append(query_features["query_emb"])
                all_query_ids[key].append(query_ids)

        losses = {}
        for key in sorted(all_query_features):
            loss = self.similarity_loss(
                query=torch.cat(all_query_features[key], dim=0),
                query_ids=torch.cat(all_query_ids[key], dim=0),
                pos=torch.cat(all_pos_features[key], dim=0),
                rnd_neg=torch.cat(all_neg_features[key], dim=0),
                pos_ids=torch.cat(all_pos_ids[key], dim=0),
                neg_ids=torch.cat(all_neg_ids[key], dim=0),
            )
            query_entity_type, candidate_entity_type, tower_state = key
            loss_key = f"{EntityType._VALUES_TO_NAMES[query_entity_type].lower()}_{EntityType._VALUES_TO_NAMES[candidate_entity_type].lower()}_{tower_state.name.lower()}"
            for wt, loss_w in loss.items():
                losses[f"{loss_key}_{wt}"] = loss_w
        assert len(all_query_features) > 0, "No valid batches"
        return {
            BasicSolver.TOTAL_LOSS: sum(losses.values()) / len(all_query_features),
            BasicSolver.LOSS_COMPONENTS: losses,
        }

    def write_summary(self, summary_writer: SummaryWriter, epoch, iteration, total_iterations):
        self.similarity_loss.write_summary(
            summary_writer=summary_writer, epoch=epoch, iteration=iteration, total_iterations=total_iterations
        )


class OmniSearchSAGEEval(OmniSearchSAGE):
    @torch.no_grad()
    def _preprocess_input(self, batch: SageBatch):
        """
        Sends all the tensors in feats.query_feats and feats.pin_feats to self.device
        """
        feats = batch.tensor_feats
        out = {}
        for entity_type, entity_feats in feats.items():
            out[entity_type] = {}
            for k, v in entity_feats.items():
                out[entity_type][k] = v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        return batch.replace(tensor_feats=out)
