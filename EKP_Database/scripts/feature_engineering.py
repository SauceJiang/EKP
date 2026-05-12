"""Generate fixed-length statistical features from cleaned DLKcat tokens."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INTERIM_CLEANED_DIR = BASE_DIR / "interim" / "cleaned"
INTERIM_FEATURE_DIR = BASE_DIR / "interim" / "statistical_features"


def stats_from_tokens(token_str: str) -> dict[str, float]:
    tokens = np.array(json.loads(token_str), dtype=float)
    return {
        "len": int(tokens.size),
        "mean": float(tokens.mean()) if tokens.size > 0 else 0.0,
        "std": float(tokens.std(ddof=0)) if tokens.size > 0 else 0.0,
        "min": float(tokens.min()) if tokens.size > 0 else 0.0,
        "max": float(tokens.max()) if tokens.size > 0 else 0.0,
        "median": float(np.median(tokens)) if tokens.size > 0 else 0.0,
    }


def main() -> None:
    INTERIM_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = INTERIM_CLEANED_DIR / "dlkcat_cleaned.csv"
    df = pd.read_csv(cleaned_path)

    protein_stats = df["protein_tokens"].apply(stats_from_tokens).apply(pd.Series)
    compound_stats = df["compound_tokens"].apply(stats_from_tokens).apply(pd.Series)

    protein_stats = protein_stats.add_prefix("protein_")
    compound_stats = compound_stats.add_prefix("compound_")

    feature_df = pd.concat([protein_stats, compound_stats, df["log10_kcat"]], axis=1)

    output_path = INTERIM_FEATURE_DIR / "dlkcat_statistical_features.csv"
    feature_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
