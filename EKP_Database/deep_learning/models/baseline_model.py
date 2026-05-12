"""Baseline multimodal sequence regressor."""

from __future__ import annotations

import torch
from torch import nn


class BaselineModel(nn.Module):
    def __init__(
        self,
        protein_encoder: nn.Module,
        compound_encoder: nn.Module,
        protein_pooling: nn.Module,
        compound_pooling: nn.Module,
        fusion: nn.Module,
        fused_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.protein_encoder = protein_encoder
        self.compound_encoder = compound_encoder
        self.protein_pooling = protein_pooling
        self.compound_pooling = compound_pooling
        self.fusion = fusion
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        compound_tokens: torch.Tensor,
        compound_mask: torch.Tensor,
    ) -> torch.Tensor:
        protein_repr = self.protein_encoder(protein_tokens)
        compound_repr = self.compound_encoder(compound_tokens)
        protein_vec = self.protein_pooling(protein_repr, protein_mask)
        compound_vec = self.compound_pooling(compound_repr, compound_mask)
        fused = self.fusion(protein_vec, compound_vec)
        return self.regressor(fused).squeeze(-1)
