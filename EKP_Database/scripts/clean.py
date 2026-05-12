"""Clean DLKcat raw arrays and write interim cleaned CSV."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_INPUT_DIR = BASE_DIR / "raw" / "DLKcat" / "input"
INTERIM_CLEANED_DIR = BASE_DIR / "interim" / "cleaned"


def to_json_list(array: np.ndarray) -> str:
    return json.dumps([int(x) for x in array])


def main() -> None:
    INTERIM_CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    proteins = np.load(RAW_INPUT_DIR / "proteins.npy", allow_pickle=True)
    compounds = np.load(RAW_INPUT_DIR / "compounds.npy", allow_pickle=True)
    regression = np.load(RAW_INPUT_DIR / "regression.npy", allow_pickle=True)

    if not (len(proteins) == len(compounds) == len(regression)):
        raise ValueError("Input arrays have inconsistent lengths.")

    protein_lengths = np.array([len(x) for x in proteins], dtype=int)
    compound_lengths = np.array([len(x) for x in compounds], dtype=int)
    target = regression.astype(float).reshape(-1)

    raw_df = pd.DataFrame(
        {
            "protein_tokens": [to_json_list(x) for x in proteins],
            "compound_tokens": [to_json_list(x) for x in compounds],
            "protein_len": protein_lengths,
            "compound_len": compound_lengths,
            "log10_kcat": target,
        }
    )

    empty_mask = (protein_lengths == 0) | (compound_lengths == 0)
    nan_mask = np.isnan(target)

    cleaned_df = raw_df.loc[~(empty_mask | nan_mask)].copy()
    cleaned_df = cleaned_df.drop_duplicates(subset=["protein_tokens", "compound_tokens", "log10_kcat"])

    out_of_range_mask = (cleaned_df["log10_kcat"] < -6.0) | (cleaned_df["log10_kcat"] > 6.0)
    cleaned_df = cleaned_df.loc[~out_of_range_mask].copy()

    cleaned_path = INTERIM_CLEANED_DIR / "dlkcat_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)


if __name__ == "__main__":
    main()
