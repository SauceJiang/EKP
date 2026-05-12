"""Inspect DLKcat raw data and write metadata reports."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_INPUT_DIR = BASE_DIR / "raw" / "DLKcat" / "input"
METADATA_DIR = BASE_DIR / "metadata"


def summarize_lengths(arr: np.ndarray) -> dict[str, float]:
    lengths = np.array([len(x) for x in arr], dtype=int)
    return {
        "count": int(lengths.size),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(median(lengths.tolist())),
    }


def main() -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    proteins = np.load(RAW_INPUT_DIR / "proteins.npy", allow_pickle=True)
    compounds = np.load(RAW_INPUT_DIR / "compounds.npy", allow_pickle=True)
    regression = np.load(RAW_INPUT_DIR / "regression.npy", allow_pickle=True)

    protein_lengths = np.array([len(x) for x in proteins], dtype=int)
    compound_lengths = np.array([len(x) for x in compounds], dtype=int)
    target = regression.astype(float).reshape(-1)

    empty_mask = (protein_lengths == 0) | (compound_lengths == 0)
    nan_mask = np.isnan(target)

    # Duplicate count based on protein+compound+target
    seen = set()
    duplicate_count = 0
    for prot, comp, y in zip(proteins, compounds, target):
        key = (tuple(int(x) for x in prot), tuple(int(x) for x in comp), float(y))
        if key in seen:
            duplicate_count += 1
        else:
            seen.add(key)

    target_min = float(np.min(target))
    target_max = float(np.max(target))
    target_range_ok = (-6.0 <= target_min <= 6.0) and (-6.0 <= target_max <= 6.0)

    shapes_md = [
        "# Dataset Shapes (DLKcat)",
        "",
        f"- proteins.npy shape: {proteins.shape}",
        f"- compounds.npy shape: {compounds.shape}",
        f"- regression.npy shape: {regression.shape}",
        "",
        "## Sequence Length Statistics",
        f"- proteins: {json.dumps(summarize_lengths(proteins))}",
        f"- compounds: {json.dumps(summarize_lengths(compounds))}",
        "",
        "## Missing and Duplicates",
        f"- empty sequence count: {int(empty_mask.sum())}",
        f"- target NaN count: {int(nan_mask.sum())}",
        f"- duplicate count: {duplicate_count}",
        "",
        "## Target Summary",
        f"- min: {target_min}",
        f"- max: {target_max}",
    ]
    (METADATA_DIR / "dataset_shapes.md").write_text("\n".join(shapes_md), encoding="utf-8")

    preprocess_md = [
        "# Preprocessing Log",
        "",
        "## Target Variable Check",
        f"- regression.npy min: {target_min}",
        f"- regression.npy max: {target_max}",
        "- conclusion: " + (
            "Assume regression.npy is already log10(kcat) because values are within [-6, 6]."
            if target_range_ok
            else "Values exceed [-6, 6]; log10 transform may be required."
        ),
        "",
        "## Notes",
        "- Token vocab semantics are not reconstructed at this stage.",
    ]
    (METADATA_DIR / "preprocessing_log.md").write_text("\n".join(preprocess_md), encoding="utf-8")


if __name__ == "__main__":
    main()
