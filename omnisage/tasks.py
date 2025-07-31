from __future__ import annotations

from typing import Dict
from typing import List
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import EntityType


class MetricLearningTask(nn.Module):
    """Simplified metric learning task for contrastive learning."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Stack positive and negative for softmax cross entropy
        # query, positive, negative: [batch, emb_dim]
        pos_sim = F.cosine_similarity(query, positive)
        neg_sim = F.cosine_similarity(query, negative)
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive at index 0
        loss = self.loss_fn(logits, labels)
        return loss

class UserSequenceTransformer(nn.Module):
    """Minimal transformer encoder for user sequence modeling."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 2, max_seq_len: int = 100, mlp_out_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.normal_(self.pos_embedding, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, mlp_out_dim),
            nn.GELU(),
            nn.Linear(mlp_out_dim, mlp_out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.mlp_head(x)
        return F.normalize(x, dim=-1)

class UserSequenceTask(nn.Module):
    """User sequence modeling with next-action and all-action retrieval objectives."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 2, max_seq_len: int = 100, mlp_out_dim: int = 128):
        super().__init__()
        self.transformer = UserSequenceTransformer(input_dim, hidden_dim, num_layers, num_heads, max_seq_len, mlp_out_dim)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # feats['sequence']: (B, T, D)
        x = feats['sequence']
        emb = self.transformer(x)  # (B, T, D)
        # Next-action retrieval: predict next embedding from current
        # For each timestep, classify the next embedding among all possible nexts
        # logits: [B, T-1, T-1], labels: [B, T-1] (correct next is always at the diagonal)
        logits = torch.bmm(emb[:, :-1, :], emb[:, 1:, :].transpose(1, 2))  # [B, T-1, T-1]
        labels = torch.arange(logits.size(1), device=logits.device).expand(logits.size(0), -1)
        next_loss = self.loss_fn(logits, labels)
        # All-action retrieval: classify each position against all others
        all_logits = torch.bmm(emb, emb.transpose(1, 2))  # [B, T, T]
        all_labels = torch.arange(all_logits.size(1), device=all_logits.device).expand(all_logits.size(0), -1)
        all_action_loss = self.loss_fn(all_logits, all_labels)
        return {'next_action_loss': next_loss, 'all_action_loss': all_action_loss}

class AutoencoderTask(nn.Module):
    """Autoencoder task for node feature reconstruction, matching embedding to projected features."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, query_os, pos_proj, neg_proj):
        # query_os: [batch, emb_dim]
        # pos_proj/neg_proj: dict of {feature: [batch, num_feats, emb_dim]}
        total_loss = 0.0
        num_feats = 0
        for feat in pos_proj:
            pos = pos_proj[feat]  # [batch, n, emb_dim]
            neg = neg_proj[feat]  # [batch, n, emb_dim]
            # For each neighbor, stack pos and neg for softmax cross entropy
            logits = torch.cat([
                F.cosine_similarity(query_os.unsqueeze(1), pos, dim=-1),
                F.cosine_similarity(query_os.unsqueeze(1), neg, dim=-1)
            ], dim=1)  # [batch, 2*n]
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)  # positive at index 0
            loss = self.loss_fn(logits, labels)
            total_loss += loss
            num_feats += 1
        if num_feats > 0:
            total_loss = total_loss / num_feats
        print(f"Total autoencoder loss: {total_loss.item()}, is_nan={torch.isnan(total_loss).item()}")
        return total_loss 