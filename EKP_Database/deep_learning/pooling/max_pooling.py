"""Max pooling with mask."""

from __future__ import annotations

import torch
from torch import nn


class MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.unsqueeze(-1)
        x = x.masked_fill(~mask, float("-inf"))
        return x.max(dim=1).values
