"""Add fusion."""

from __future__ import annotations

import torch
from torch import nn


class AddFusion(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
