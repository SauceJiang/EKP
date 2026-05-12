"""PyTorch dataset for DLKcat token sequences."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(self, raw_dir: Path) -> None:
        self.proteins = np.load(raw_dir / "proteins.npy", allow_pickle=True)
        self.compounds = np.load(raw_dir / "compounds.npy", allow_pickle=True)
        self.targets = np.load(raw_dir / "regression.npy", allow_pickle=True).astype(float).reshape(-1)

        if not (len(self.proteins) == len(self.compounds) == len(self.targets)):
            raise ValueError("Input arrays have inconsistent lengths.")

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {
            "protein": self.proteins[idx].tolist(),
            "compound": self.compounds[idx].tolist(),
            "target": float(self.targets[idx]),
        }

    def vocab_sizes(self) -> Dict[str, int]:
        protein_max = int(np.max([np.max(x) for x in self.proteins]))
        compound_max = int(np.max([np.max(x) for x in self.compounds]))
        return {"protein": protein_max + 1, "compound": compound_max + 1}
