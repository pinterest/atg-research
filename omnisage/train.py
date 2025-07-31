from __future__ import annotations

from typing import Dict

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GraphSageV6AutoencoderEmbedder
from tasks import AutoencoderTask
from tasks import MetricLearningTask
from tasks import UserSequenceTask

# Define self and neighbor feature dimensions
SELF_FEATURE_DIMS = {
    "ue_v4": 1024,
    "sig_annotations_v5_k50_terms": 128,
    "self_degree": 1,
}
NEIGHBOR_FEATURE_DIMS = {
    # pin_board graph
    "neigh_pin_board_ue_v4": 1024,
    "neigh_pin_board_sig_annotations_v5_k50_terms": 128,
    "neigh_pin_board_self_degree": 1,
    "neigh_pin_board_board_annotations_v5_terms": 128,  # board neighbors
    # pin_pin graph
    "neigh_pin_pin_ue_v4": 1024,
    "neigh_pin_pin_sig_annotations_v5_k50_terms": 128,
    "neigh_pin_pin_self_degree": 1,
}

# Set neighbor counts for each graph/entity
GRAPH_TO_NUM_NEIGHBORS = {
    'pin_board': {0: 25, 1: 10},  # 0: SIGNATURE, 1: BOARD
    'pin_pin': {0: 15},
}

# Helper to get neighbor count for each feature
NEIGHBOR_COUNTS = {
    "neigh_pin_board_ue_v4": 25,
    "neigh_pin_board_sig_annotations_v5_k50_terms": 25,
    "neigh_pin_board_self_degree": 25,
    "neigh_pin_board_board_annotations_v5_terms": 10,
    "neigh_pin_pin_ue_v4": 15,
    "neigh_pin_pin_sig_annotations_v5_k50_terms": 15,
    "neigh_pin_pin_self_degree": 15,
}

def generate_fake_pin_features(batch_size: int = 32) -> Dict[str, torch.Tensor]:
    feats = {}
    for feat_name, dim in SELF_FEATURE_DIMS.items():
        if feat_name.endswith("degree"):
            feats[feat_name] = torch.rand(batch_size, dim) * 10
        else:
            feats[feat_name] = torch.randn(batch_size, dim).clamp(-5, 5)
    for feat_name, dim in NEIGHBOR_FEATURE_DIMS.items():
        num_neighbors = NEIGHBOR_COUNTS[feat_name]
        if feat_name.endswith("degree"):
            feats[feat_name] = torch.rand(batch_size, num_neighbors, dim) * 10
        else:
            feats[feat_name] = torch.randn(batch_size, num_neighbors, dim).clamp(-5, 5)
    feats['neighbor_weight_pin_board_signature'] = torch.rand(batch_size, 25) + 1e-3
    feats['neighbor_weight_pin_board_board'] = torch.rand(batch_size, 10) + 1e-3
    feats['neighbor_weight_pin_pin_signature'] = torch.rand(batch_size, 15) + 1e-3
    return feats

def create_model() -> nn.Module:
    model = GraphSageV6AutoencoderEmbedder(
        self_feature_dims=SELF_FEATURE_DIMS,
        neighbor_feature_dims=NEIGHBOR_FEATURE_DIMS,
        emb_size=128,
    )
    return model

def train_step(
    model: nn.Module,
    auto_loss_fn: nn.Module,
    metric_loss_fn: nn.Module,
    user_seq_task: nn.Module,
    optimizer: optim.Optimizer,
    batch: Dict[str, Dict[str, torch.Tensor]],
    seq_feats: Dict[str, torch.Tensor],
    device: torch.device,
    seq_len: int = 256,
) -> tuple:
    # batch: {'query': ..., 'positive': ..., 'negative': ...}
    for k in batch:
        batch[k] = {kk: vv.to(device) for kk, vv in batch[k].items()}
    optimizer.zero_grad()
    # Model forward for each group
    query_os, query_proj = model(batch['query'])
    pos_os, pos_proj = model(batch['positive'])
    neg_os, neg_proj = model(batch['negative'])
    # Metric learning: use OS embeddings
    metric_loss = metric_loss_fn(query_os, pos_os, neg_os)
    # Autoencoder contrastive: use OS embedding of query, projected features of pos/neg
    auto_loss = auto_loss_fn(query_os, pos_proj, neg_proj)
    # --- User sequence task ---
    embedded_sequence = embed_sequence_with_omnisage(model, seq_feats, device)  # [BS, seq_len, 128]
    user_seq_losses = user_seq_task({'sequence': embedded_sequence.to(device)})
    user_seq_next_loss = user_seq_losses['next_action_loss']
    user_seq_all_loss = user_seq_losses['all_action_loss']
    # Combine all losses
    total_loss = metric_loss + auto_loss + user_seq_next_loss + user_seq_all_loss
    total_loss.backward()
    optimizer.step()
    return metric_loss.item(), auto_loss.item(), user_seq_next_loss.item(), user_seq_all_loss.item()

def generate_metric_learning_data(
    batch_size: int = 32,
) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        'query': generate_fake_pin_features(batch_size),
        'positive': generate_fake_pin_features(batch_size),
        'negative': generate_fake_pin_features(batch_size),
    }

def generate_sequence_dataset(batch_size: int = 32, seq_len: int = 256) -> Dict[str, torch.Tensor]:
    flat_feats = generate_fake_pin_features(batch_size * seq_len)
    seq_feats = {}
    for k, v in flat_feats.items():
        # v: [BS*seq_len, ...] -> [BS, seq_len, ...]
        if v.dim() == 2:
            seq_feats[k] = v.view(batch_size, seq_len, v.shape[-1])
        elif v.dim() == 3:
            seq_feats[k] = v.view(batch_size, seq_len, v.shape[1], v.shape[2])
        else:
            raise ValueError(f"Unexpected feature dim for {k}: {v.shape}")
    return seq_feats

def embed_sequence_with_omnisage(model, seq_feats, device):
    batch_size, seq_len = next(iter(seq_feats.values())).shape[:2]
    # Flatten to [BS*seq_len, ...] for each feature
    flat_feats = {}
    for k, v in seq_feats.items():
        if v.dim() == 3:
            flat_feats[k] = v.view(batch_size * seq_len, v.shape[-1]).to(device)
        elif v.dim() == 4:
            flat_feats[k] = v.reshape(batch_size * seq_len, v.shape[2], v.shape[3]).to(device)
        else:
            raise ValueError(f"Unexpected feature dim for {k}: {v.shape}")
    # Embed each pin
    with torch.no_grad():
        emb, _ = model(flat_feats)
    # emb: [BS*seq_len, 128] -> [BS, seq_len, 128]
    emb = emb.view(batch_size, seq_len, -1)
    return emb

def train_metric_learning_step(
    task: nn.Module,
    optimizer: optim.Optimizer,
    feats: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device
) -> float:
    feats = {k: {kk: vv.to(device) for kk, vv in v.items()} for k, v in feats.items()}
    optimizer.zero_grad()
    loss = task(feats)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = create_model().to(device)
    auto_loss_fn = AutoencoderTask()
    metric_loss_fn = MetricLearningTask()
    num_epochs = 5
    batch_size = 8
    num_batches = 10
    seq_len = 256
    user_seq_task = UserSequenceTask(input_dim=128, max_seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print("Starting joint training (autoencoder + metric learning + user sequence)...")
    for epoch in range(num_epochs):
        metric_epoch_loss = 0.0
        auto_epoch_loss = 0.0
        user_seq_epoch_next_loss = 0.0
        user_seq_epoch_all_loss = 0.0
        for batch_idx in range(num_batches):
            batch = generate_metric_learning_data(batch_size=batch_size)
            seq_feats = generate_sequence_dataset(batch_size, seq_len)
            metric_loss, auto_loss, user_seq_next_loss, user_seq_all_loss = train_step(
                model, auto_loss_fn, metric_loss_fn, user_seq_task, optimizer, batch, seq_feats, device, seq_len
            )
            metric_epoch_loss += metric_loss
            auto_epoch_loss += auto_loss
            user_seq_epoch_next_loss += user_seq_next_loss
            user_seq_epoch_all_loss += user_seq_all_loss
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{num_batches}, MetricLoss: {metric_loss:.4f}, AutoLoss: {auto_loss:.4f}, UserSeqNextLoss: {user_seq_next_loss:.4f}, UserSeqAllLoss: {user_seq_all_loss:.4f}")
        avg_metric_loss = metric_epoch_loss / num_batches
        avg_auto_loss = auto_epoch_loss / num_batches
        avg_user_seq_next_loss = user_seq_epoch_next_loss / num_batches
        avg_user_seq_all_loss = user_seq_epoch_all_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} completed. Avg MetricLoss: {avg_metric_loss:.4f}, Avg AutoLoss: {avg_auto_loss:.4f}, Avg UserSeqNextLoss: {avg_user_seq_next_loss:.4f}, Avg UserSeqAllLoss: {avg_user_seq_all_loss:.4f}")
    print("Joint training completed!")

if __name__ == "__main__":
    main() 