"""Model evaluation helper."""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .metrics import mae, r2, rmse


def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            protein = batch["protein"].to(device)
            protein_mask = batch["protein_mask"].to(device)
            compound = batch["compound"].to(device)
            compound_mask = batch["compound_mask"].to(device)
            target = batch["target"].to(device)

            output = model(protein, protein_mask, compound, compound_mask)
            y_true.append(target.cpu().numpy())
            y_pred.append(output.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
    }
