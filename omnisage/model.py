from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import copy
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class EntityType:
    SIGNATURE = 0
    BOARD = 1
    _VALUES_TO_NAMES = {0: "SIGNATURE", 1: "BOARD"}


class MultiVocabTokenizer:
    """Simple tokenizer for text features."""
    @staticmethod
    def default(vocab_type: str = "all") -> Dict[str, int]:
        """Returns a default vocabulary mapping."""
        # This is a simplified version - in practice you'd want a proper vocabulary representative of the text features
        return {"<pad>": 0, "<unk>": 1, "<cls>": 2, "<sep>": 3}

class HashEmbeddingBag(nn.Module):
    """Hash-based embedding bag for text features."""

    def __init__(
        self,
        num_hashes: int = 2,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        num_embeds: int = 100_000,
        hash_weights: bool = False,
        normalize: bool = True,
    ):
        super().__init__()
        self.num_hashes = num_hashes
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_embeds = num_embeds
        self.hash_weights = hash_weights
        self.normalize = normalize

        # Initialize embedding table
        self.embeddings = nn.Parameter(torch.Tensor(num_embeds, embedding_dim))
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, input_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
            offsets: Tensor of shape (batch_size,) containing sequence offsets

        Returns:
            Tensor of shape (batch_size, embedding_dim) containing text embeddings
        """
        # Hash the input IDs
        hashed_ids = input_ids % self.num_embeds

        # Get embeddings
        embeds = self.embeddings[hashed_ids]

        # Sum embeddings for each sequence
        batch_size = offsets.shape[0]
        output = torch.zeros(batch_size, self.embedding_dim, device=embeds.device)

        for i in range(batch_size):
            start = offsets[i]
            end = offsets[i + 1] if i < batch_size - 1 else input_ids.shape[1]
            output[i] = embeds[i, start:end].sum(dim=0)

        # Normalize if requested
        if self.normalize:
            output = F.normalize(output, p=2, dim=-1)
        return output

class PinTextEmbedder(nn.Module):
    """Text embedder for pin features."""
    
    def __init__(self, text_embedding_dim: int = 512):
        super().__init__()
        
        self.text_embedder = HashEmbeddingBag(
            num_hashes=2,
            vocab_size=len(MultiVocabTokenizer.default("all")),
            embedding_dim=text_embedding_dim,
            num_embeds=100_000,
            hash_weights=False,
            normalize=True,
        )
        
    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            feats: Dictionary containing text features with keys ending in '_input_ids' and '_offsets'
            
        Returns:
            Tensor of shape (batch_size, num_features, embedding_dim) containing text embeddings
        """
        text_embs: List[torch.Tensor] = []
        
        # Process each text feature
        for feature_name in sorted(feats.keys()):
            if feature_name.endswith("_input_ids"):
                key = feature_name[: -len("_input_ids")]
                text_embs.append(
                    self.text_embedder(
                        feats[f"{key}_input_ids"],
                        feats[f"{key}_offsets"]
                    )
                )
                
        return torch.stack(text_embs, dim=1)

class FeatureProjectionLayer(nn.Module):
    """Projects multiple feature embeddings using a simple linear layer and activation function."""
    
    def __init__(
        self,
        dimensions_dict: Dict[str, int],
        output_dim: int,
        input_dim: Optional[int] = None,
        activation_fn: Optional[nn.Module] = nn.GELU,
    ):
        super().__init__()
        if input_dim:
            self.linear_dict = nn.ModuleDict({key: nn.Linear(input_dim, output_dim) for key in dimensions_dict})
        else:
            self.linear_dict = nn.ModuleDict({key: nn.Linear(val, output_dim) for key, val in dimensions_dict.items()})
        self.act = activation_fn() if activation_fn else nn.Identity()

    def forward(self, tensor_descriptions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {desc: self.act(self.linear_dict[desc](tensor)) for desc, tensor in tensor_descriptions.items()}

class TransformerAggregator(nn.Module):
    """Simple transformer-based aggregator for graph features."""
    
    def __init__(
        self,
        sequence_length: int = None,
        heads: int = 12,
        hidden_size: int = 768,
        dropout: float = 0.0,
        mlp_size: int = 3072,
        num_layers: int = 4,
        emb_size: int = 256,
        num_output_heads: int = 1,
        export: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_output_heads = num_output_heads
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=heads,
                dim_feedforward=mlp_size,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        # Output projection
        self.output_proj = nn.Linear(hidden_size, emb_size)
        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, sequence_length + num_output_heads, hidden_size))
        nn.init.normal_(self.pos_embedding, std=0.02)
        # Global token
        self.global_token = nn.Parameter(torch.zeros(1, num_output_heads, hidden_size))
        nn.init.normal_(self.global_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add global token
        batch_size = x.size(0)
        global_token = self.global_token.expand(batch_size, -1, -1)
        x = torch.cat([global_token, x], dim=1)
        # Add positional embedding
        x = x + self.pos_embedding[:, :x.size(1), :]
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        # Project to output dimension
        x = self.output_proj(x)
        # Return only the first token (global token) embedding
        if self.num_output_heads == 1:
            return x[:, 0]
        else:
            return x[:, :self.num_output_heads]

class GraphSageV6AlphaTransformerAggregator(nn.Module):
    """Unified transformer aggregator for GraphSage v6, supports both standard and autoencoder modes."""
    def __init__(
        self,
        graph_to_num_neighbors: Dict[str, Dict[EntityType, int]] = None,
        entity_to_feature_names: Dict[EntityType, List[str]] = None,
        self_feature_dims: Dict[str, int] = None,
        neighbor_feature_dims: Dict[str, int] = None,
        heads: int = 12,
        hidden_size: int = 768,
        dropout: float = 0.0,
        mlp_size: int = 3072,
        num_layers: int = 4,
        emb_size: int = 256,
        export: bool = False,
    ):
        super().__init__()
        self.graph_to_num_neighbors = graph_to_num_neighbors
        self.entity_to_feature_names = entity_to_feature_names
        self.self_feature_dims = self_feature_dims
        self.neighbor_feature_dims = neighbor_feature_dims
        self.emb_size = emb_size
        # Feature projection layers for self features
        if self_feature_dims is not None:
            self.self_proj = nn.ModuleDict({k: nn.Linear(v, emb_size) for k, v in self_feature_dims.items()})
        # Feature projection layers for neighbor features
        if neighbor_feature_dims is not None:
            self.neigh_proj = nn.ModuleDict({k: nn.Linear(v, emb_size) for k, v in neighbor_feature_dims.items()})
        # For legacy/standard mode
        self.self_visual_mlp = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(2048, hidden_size)
        )
        self.self_annotations_mlp = nn.Sequential(
            nn.Linear(128, 2048), nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(2048, hidden_size)
        )
        self.self_degree_linear = nn.Linear(1, hidden_size)
        self.neigh_visual_mlp = nn.Sequential(
            nn.Linear(1024, 2048), nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(2048, hidden_size // 2)
        )
        self.neigh_annotations_mlp = nn.Sequential(
            nn.Linear(128, 2048), nn.ReLU(), nn.Dropout(p=0.25), nn.Linear(2048, hidden_size // 2)
        )
        self.neigh_weight_linear = nn.Linear(1, hidden_size)
        self.all_text_linear = nn.Linear(512, hidden_size)
        # Transformer aggregator
        total_neighbors = 0
        if self.graph_to_num_neighbors is not None:
            total_neighbors = sum(
                [sum(graph_neighbors.values()) for graph_neighbors in self.graph_to_num_neighbors.values()]
            )
        self.aggregator = TransformerAggregator(
            sequence_length=3 + total_neighbors,
            heads=heads,
            hidden_size=hidden_size,
            dropout=dropout,
            mlp_size=mlp_size,
            num_layers=num_layers,
            emb_size=emb_size,
            num_output_heads=1,
            export=export,
        )
        # For autoencoder: final aggregation on pooled embedding
        self.agg = nn.Linear(emb_size, emb_size)

    def extract_neigh_embs_from_feats(self, feats: Dict[str, torch.Tensor]) -> Dict[str, Dict[int, torch.Tensor]]:
        """
        Extracts neighbor annotation features from feats using the 'neigh_' prefix and organizes them as neigh_embs[graph][entity].
        """
        neigh_embs = {}
        for graph in (self.graph_to_num_neighbors or {}):
            neigh_embs[graph] = {}
            for entity in self.graph_to_num_neighbors[graph]:
                # Try to find annotation features for this graph/entity
                # e.g., neigh_pin_board_sig_annotations_v5_k50_terms or neigh_pin_board_board_annotations_v5_terms
                for feat_name in (self.entity_to_feature_names or {}).get(entity, []):
                    if "annotations" in feat_name:
                        key = f"neigh_{graph}_{feat_name}"
                        if key in feats:
                            neigh_embs[graph][entity] = feats[key]
        return neigh_embs

    def process_self_features(
        self,
        feats: Dict[str, torch.Tensor],
        return_raw_feats: bool = False,
        current_entity_type: EntityType = EntityType.SIGNATURE,
    ) -> torch.Tensor:
        """Process self features (visual, annotations, degree)."""
        self_visual_proc = self.self_visual_mlp(feats["ue_v4"]).unsqueeze(1)
        # Extract self_annotations here
        self_annotations = feats.get('sig_annotations_v5_k50_terms', None)
        self_annotations_proc = self.self_annotations_mlp(self_annotations).unsqueeze(1)
        self_degree_proc = self.self_degree_linear(torch.log(1 + feats["self_degree"])).unsqueeze(1)

        processed_features = [self_visual_proc, self_annotations_proc, self_degree_proc]

        if return_raw_feats:
            return processed_features, (
                [feats["ue_v4"], self_annotations, torch.log(1 + feats["self_degree"])],
                ["self_visual", "self_annotations", "self_degree"],
            )
        return processed_features

    def process_neighbor_features(
        self,
        neigh_embs: Dict[str, torch.Tensor],
        feats: Dict[str, torch.Tensor],
        return_raw_feats: bool = False,
        current_entity: EntityType = EntityType.SIGNATURE,
    ):
        """Process neighbor features."""
        processed_features = []
        for graph, entity_to_num_neigh in self.graph_to_num_neighbors.items():
            if current_entity not in entity_to_num_neigh:
                continue
                
            for entity, num_neigh in entity_to_num_neigh.items():
                # Process visual features
                visual_feat = self.neigh_visual_mlp(feats[f"neigh_{graph}_ue_v4"][:, :num_neigh])
                
                # Process annotations
                if entity in neigh_embs[graph]:
                    annotations_feat = self.neigh_annotations_mlp(neigh_embs[graph][entity][:, :num_neigh])
                else:
                    annotations_feat = torch.zeros_like(visual_feat)
                
                # Process weights
                weight_feat = self.neigh_weight_linear(
                    torch.log(1 + feats[f"neighbor_weight_{graph}_{EntityType._VALUES_TO_NAMES[entity].lower()}"][:, :num_neigh].unsqueeze(2))
                )
                
                # Combine features
                combined = torch.cat([visual_feat, annotations_feat], dim=2) * weight_feat
                processed_features.append(combined)
                
        return processed_features

    def forward(
        self,
        feats: Dict[str, torch.Tensor] = None,
        neigh_embs: Dict[str, torch.Tensor] = None,
        all_text: torch.Tensor = None,
        current_entity_type: EntityType = EntityType.SIGNATURE,
    ):
        # --- Compute OmniSage embedding (normal pipeline) ---
        if neigh_embs is None:
            neigh_embs = self.extract_neigh_embs_from_feats(feats)
        processed_features = self.process_self_features(feats, False, current_entity_type)
        processed_features.extend(
            self.process_neighbor_features(neigh_embs, feats, False, current_entity=current_entity_type)
        )
        if all_text is not None:
            all_text = self.all_text_linear(all_text)
            processed_features.append(all_text)
        processed_features = [f if f.dim() == 3 else f.unsqueeze(1) for f in processed_features]
        all_embs = torch.cat(processed_features, dim=1)
        os_embedding = F.normalize(self.aggregator(all_embs), dim=-1)

        # --- Compute projections dict (autoencoder pipeline) ---
        self_proj = {k: F.normalize(self.self_proj[k](feats[k]), dim=-1) for k in self.self_feature_dims if k in feats}
        neigh_proj = {k: F.normalize(self.neigh_proj[k](feats[k]), dim=-1) for k in self.neighbor_feature_dims if k in feats}
        projections_dict = {**self_proj, **neigh_proj}
        return os_embedding, projections_dict

class GraphSageV6AutoencoderEmbedder(nn.Module):
    """Embedder for autoencoder: prepares self and neighbor features and calls aggregator."""
    def __init__(self, self_feature_dims: Dict[str, int], neighbor_feature_dims: Dict[str, int], emb_size: int = 128):
        super().__init__()
        # Use the same structure as in train.py
        graph_to_num_neighbors = {
            'pin_board': {EntityType.SIGNATURE: 25, EntityType.BOARD: 10},
            'pin_pin': {EntityType.SIGNATURE: 15},
        }
        entity_to_feature_names = {
            EntityType.SIGNATURE: [
                "ue_v4",
                "sig_annotations_v5_k50_terms",
                "self_degree",
            ],
            EntityType.BOARD: [
                "board_annotations_v5_terms",
            ],
        }
        self.aggregator = GraphSageV6AlphaTransformerAggregator(
            graph_to_num_neighbors=graph_to_num_neighbors,
            entity_to_feature_names=entity_to_feature_names,
            self_feature_dims=self_feature_dims,
            neighbor_feature_dims=neighbor_feature_dims,
            emb_size=emb_size,
        )

    def forward(self, feats: Dict[str, torch.Tensor], **kwargs) -> tuple:
        # Always extract neigh_embs from feats using the aggregator's method
        return self.aggregator(feats=feats) 