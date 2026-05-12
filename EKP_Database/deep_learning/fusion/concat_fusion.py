"""Concatenate fusion."""

from __future__ import annotations

import torch
from torch import nn


class ConcatFusion(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b], dim=-1)
