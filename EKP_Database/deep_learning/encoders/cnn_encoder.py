"""CNN encoder for token sequences."""

from __future__ import annotations

import torch
from torch import nn


class CNNEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)  # (B, L, D)
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.conv(x)
        x = self.activation(x)
        x = x.transpose(1, 2)  # (B, L, H)
        return self.dropout(x)
