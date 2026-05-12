"""Batch collation with padding and truncation."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _pad_truncate(seq: List[int], max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    seq = seq[:max_len]
    pad_len = max_len - len(seq)
    padded = torch.tensor(seq + [0] * pad_len, dtype=torch.long)
    mask = torch.tensor([1] * len(seq) + [0] * pad_len, dtype=torch.bool)
    return padded, mask


def collate_batch(batch: List[Dict[str, List[int]]], protein_max_len: int, compound_max_len: int):
    protein_batch = []
    protein_mask = []
    compound_batch = []
    compound_mask = []
    targets = []

    for item in batch:
        p, p_mask = _pad_truncate(item["protein"], protein_max_len)
        c, c_mask = _pad_truncate(item["compound"], compound_max_len)
        protein_batch.append(p)
        protein_mask.append(p_mask)
        compound_batch.append(c)
        compound_mask.append(c_mask)
        targets.append(item["target"])

    return {
        "protein": torch.stack(protein_batch),
        "protein_mask": torch.stack(protein_mask),
        "compound": torch.stack(compound_batch),
        "compound_mask": torch.stack(compound_mask),
        "target": torch.tensor(targets, dtype=torch.float32),
    }
