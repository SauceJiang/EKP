"""Run EDA, visualization, and correlation analysis for DLKcat features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
FEATURE_PATH = BASE_DIR / "interim" / "statistical_features" / "dlkcat_statistical_features.csv"
RESULTS_DIR = BASE_DIR / "results" / "eda"

TARGET_COL = "log10_kcat"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(FEATURE_PATH)
    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Target distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(df[TARGET_COL], bins=40, kde=True)
    plt.title("log10(kcat) Distribution")
    plt.xlabel("log10(kcat)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "target_hist.png", dpi=150)
    plt.close()

    # Feature distributions
    for col in feature_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], bins=40, kde=False)
        plt.title(f"Histogram: {col}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"feature_hist_{col}.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 2.5))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot: {col}")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"feature_box_{col}.png", dpi=150)
        plt.close()

    # Correlation matrix and heatmap
    corr = df.corr(numeric_only=True)
    corr.to_csv(RESULTS_DIR / "correlation_analysis.csv", index=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Pearson Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Correlation with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=lambda x: x.abs(), ascending=False)
    top_features = target_corr.head(10)

    summary_lines = [
        "# EDA Summary",
        "",
        "## Target Summary",
        df[TARGET_COL].describe().to_string(),
        "",
        "## Top Correlations with Target",
    ]
    for name, value in top_features.items():
        summary_lines.append(f"- {name}: {value:.4f}")

    summary_lines.append("")
    summary_lines.append("## Notes")
    summary_lines.append("- Correlations are Pearson coefficients based on statistical features.")
    summary_lines.append("- Features with |r| >= 0.3 are considered potentially useful predictors.")

    (RESULTS_DIR / "eda_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
