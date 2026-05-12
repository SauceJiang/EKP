"""Split statistical feature table into train/valid/test sets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
INTERIM_FEATURE_DIR = BASE_DIR / "interim" / "statistical_features"
PROCESSED_DIR = BASE_DIR / "processed"

RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1


def main() -> None:
    (PROCESSED_DIR / "train").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "valid").mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "test").mkdir(parents=True, exist_ok=True)

    feature_path = INTERIM_FEATURE_DIR / "dlkcat_statistical_features.csv"
    df = pd.read_csv(feature_path)

    rng = np.random.default_rng(RANDOM_SEED)
    indices = rng.permutation(len(df))
    train_end = int(len(indices) * TRAIN_RATIO)
    valid_end = train_end + int(len(indices) * VALID_RATIO)

    train_idx = indices[:train_end]
    valid_idx = indices[train_end:valid_end]
    test_idx = indices[valid_end:]

    df.iloc[train_idx].to_csv(PROCESSED_DIR / "train" / "dlkcat_train.csv", index=False)
    df.iloc[valid_idx].to_csv(PROCESSED_DIR / "valid" / "dlkcat_valid.csv", index=False)
    df.iloc[test_idx].to_csv(PROCESSED_DIR / "test" / "dlkcat_test.csv", index=False)


if __name__ == "__main__":
    main()
