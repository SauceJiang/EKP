"""Token embedding encoder."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.embedding(tokens))
