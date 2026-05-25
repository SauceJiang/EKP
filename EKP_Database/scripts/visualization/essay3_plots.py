"""Generate Essay3 EDA figures from processed EKP datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parent

CLEANED_PATH = BASE_DIR / "interim" / "cleaned" / "dlkcat_cleaned.csv"
TRAIN_PATH = BASE_DIR / "processed" / "train" / "dlkcat_train.csv"
VALID_PATH = BASE_DIR / "processed" / "valid" / "dlkcat_valid.csv"
TEST_PATH = BASE_DIR / "processed" / "test" / "dlkcat_test.csv"

FIGURES_DIR = REPO_ROOT / "Title" / "Essay" / "figures" / "essay3"


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )


def gaussian_kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return np.zeros_like(grid)

    std = float(np.std(values, ddof=1)) if n > 1 else 0.0
    bandwidth = 1.06 * std * (n ** (-1.0 / 5.0)) if std > 0 else 1.0
    bandwidth = max(bandwidth, 1e-3)

    diffs = (grid[:, None] - values[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diffs ** 2)
    density = kernel.mean(axis=1) / (bandwidth * np.sqrt(2 * np.pi))
    return density


def save_histogram(values: np.ndarray, bins: int, title: str, xlabel: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.0))
    plt.hist(values, bins=bins, color="#4C72B0", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_target_distribution(values: np.ndarray, bins: int, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 4.0))
    counts, bin_edges, _ = plt.hist(
        values,
        bins=bins,
        color="#4C72B0",
        edgecolor="white",
        alpha=0.85,
    )

    bin_width = bin_edges[1] - bin_edges[0]
    grid = np.linspace(bin_edges[0], bin_edges[-1], 300)
    density = gaussian_kde_1d(values, grid)
    kde_scaled = density * values.size * bin_width

    plt.plot(grid, kde_scaled, color="#DD8452", linewidth=2)
    plt.title("log10(kcat) Target Distribution")
    plt.xlabel("log10(kcat)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_boxplot(values: np.ndarray, title: str, xlabel: str, out_path: Path) -> None:
    plt.figure(figsize=(6.5, 2.8))
    plt.boxplot(values, vert=False, patch_artist=True, boxprops={"facecolor": "#55A868"})
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_split_bar(train_count: int, valid_count: int, test_count: int, out_path: Path) -> None:
    plt.figure(figsize=(6.0, 4.0))
    labels = ["Train", "Validation", "Test"]
    counts = [train_count, valid_count, test_count]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    plt.bar(labels, counts, color=colors)
    plt.title("Dataset Split Counts")
    plt.xlabel("Split")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    setup_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cleaned = pd.read_csv(CLEANED_PATH)
    protein_lengths = cleaned["protein_len"].to_numpy()
    compound_lengths = cleaned["compound_len"].to_numpy()
    target = cleaned["log10_kcat"].to_numpy()

    train_count = len(pd.read_csv(TRAIN_PATH))
    valid_count = len(pd.read_csv(VALID_PATH))
    test_count = len(pd.read_csv(TEST_PATH))

    save_histogram(
        protein_lengths,
        bins=50,
        title="Protein Sequence Length Distribution",
        xlabel="Protein sequence length",
        out_path=FIGURES_DIR / "protein_length_distribution.png",
    )

    save_histogram(
        compound_lengths,
        bins=50,
        title="Compound Token Length Distribution",
        xlabel="Compound token length",
        out_path=FIGURES_DIR / "compound_length_distribution.png",
    )

    save_target_distribution(
        target,
        bins=40,
        out_path=FIGURES_DIR / "log10_kcat_distribution.png",
    )

    save_split_bar(
        train_count,
        valid_count,
        test_count,
        out_path=FIGURES_DIR / "dataset_split_counts.png",
    )

    save_boxplot(
        protein_lengths,
        title="Protein Length Boxplot",
        xlabel="Protein sequence length",
        out_path=FIGURES_DIR / "protein_length_boxplot.png",
    )

    save_boxplot(
        compound_lengths,
        title="Compound Length Boxplot",
        xlabel="Compound token length",
        out_path=FIGURES_DIR / "compound_length_boxplot.png",
    )


if __name__ == "__main__":
    main()
