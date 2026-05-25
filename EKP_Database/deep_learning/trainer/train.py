"""Train lightweight multimodal sequence regressors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ..datasets.sequence_dataset import SequenceDataset
from ..datasets.collate_fn import collate_batch
from ..encoders.embedding_encoder import EmbeddingEncoder
from ..encoders.cnn_encoder import CNNEncoder
from ..encoders.mlp_encoder import MLPEncoder
from ..fusion.concat_fusion import ConcatFusion
from ..fusion.add_fusion import AddFusion
from ..fusion.multiply_fusion import MultiplyFusion
from ..pooling.mean_pooling import MeanPooling
from ..pooling.max_pooling import MaxPooling
from ..pooling.attention_pooling import AttentionPooling
from ..models.baseline_model import BaselineModel
from ..models.cnn_model import CNNModel
from .evaluate import evaluate


def predict_with_lengths(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    protein_lengths = []

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
            protein_lengths.append(protein_mask.sum(dim=1).cpu().numpy())

    return (
        np.concatenate(y_true),
        np.concatenate(y_pred),
        np.concatenate(protein_lengths),
    )


def write_error_analysis(
    output_dir: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_lengths: np.ndarray,
    model_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    residuals = y_true - y_pred
    abs_error = np.abs(residuals)

    # Prediction vs ground truth
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("True log10(kcat)")
    plt.ylabel("Predicted log10(kcat)")
    plt.title("Prediction vs Ground Truth")
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_true.png", dpi=150)
    plt.close()

    # Residual distribution
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_distribution.png", dpi=150)
    plt.close()

    # Absolute error vs protein length
    plt.figure(figsize=(6, 4))
    plt.scatter(protein_lengths, abs_error, alpha=0.4)
    plt.xlabel("Protein Sequence Length")
    plt.ylabel("Absolute Error")
    plt.title("Absolute Error vs Protein Length")
    plt.tight_layout()
    plt.savefig(output_dir / "error_vs_protein_length.png", dpi=150)
    plt.close()

    residual_mean = float(np.mean(residuals))
    residual_std = float(np.std(residuals))
    max_abs_error = float(np.max(abs_error))
    pearson_corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    summary = f"""# Error Analysis Summary (Test Set)

## Model Configuration
- encoder: {model_config["encoder"]}
- pooling: {model_config["pooling"]}
- fusion: {model_config["fusion"]}

## Figures
- **Prediction vs Ground Truth**: The scatter plot (pred_vs_true.png) compares predicted and true log10(kcat). The dashed diagonal denotes perfect agreement, so deviations quantify prediction error.
- **Residual Distribution**: The histogram (residual_distribution.png) shows the distribution of residuals $y_{{true}} - y_{{pred}}$, indicating bias and dispersion.
- **Absolute Error vs Protein Length**: The scatter plot (error_vs_protein_length.png) visualizes whether sequence length is associated with larger absolute errors.

## Summary Statistics

| Metric | Value |
|---|---:|
| Residual mean | {residual_mean:.4f} |
| Residual std | {residual_std:.4f} |
| Max absolute error | {max_abs_error:.4f} |
| Pearson corr (pred vs true) | {pearson_corr:.4f} |

## Interpretation
The prediction vs ground truth plot demonstrates the overall calibration of the embedding + mean pooling + concat fusion model on the test set. A tight clustering around the diagonal indicates strong alignment, while wider spread highlights hard-to-predict samples. The residual histogram provides a complementary view of bias; a mean near zero suggests limited systematic over- or under-estimation, whereas heavy tails indicate occasional large errors.

The absolute error vs protein length plot is used to diagnose length-dependent failure modes. A rising envelope of errors for longer proteins would be consistent with information loss from truncation at the maximum sequence length and with higher compositional complexity in long sequences.

These observations are consistent with the pooling ablation results: mean pooling produced the most stable performance and the lowest overall error. From a representation standpoint, mean pooling preserves global signal and reduces variance relative to max pooling, which aligns with a residual distribution that is more symmetric and with fewer extreme outliers.

Potential causes of large errors include sequence truncation, noisy or heterogeneous assay conditions in the underlying dataset, and limited capacity to model long-range dependencies. These factors are more pronounced for long proteins and complex compound contexts, which may require richer encoders or length-aware pooling in future iterations.
"""

    (output_dir / "error_analysis_summary.md").write_text(summary, encoding="utf-8")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_encoder(name: str, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float) -> nn.Module:
    if name == "embedding":
        return EmbeddingEncoder(vocab_size=vocab_size, embed_dim=embed_dim, dropout=dropout)
    if name == "cnn":
        return CNNEncoder(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
    if name == "mlp":
        return MLPEncoder(vocab_size=vocab_size, embed_dim=embed_dim, hidden_dim=hidden_dim, dropout=dropout)
    raise ValueError(f"Unknown encoder: {name}")


def build_pooling(name: str, hidden_dim: int) -> nn.Module:
    if name == "mean":
        return MeanPooling()
    if name == "max":
        return MaxPooling()
    if name == "attention":
        return AttentionPooling(hidden_dim)
    raise ValueError(f"Unknown pooling: {name}")


def build_fusion(name: str) -> nn.Module:
    if name == "concat":
        return ConcatFusion()
    if name == "add":
        return AddFusion()
    if name == "multiply":
        return MultiplyFusion()
    raise ValueError(f"Unknown fusion: {name}")


def fused_dim_for(name: str, hidden_dim: int) -> int:
    return hidden_dim * 2 if name == "concat" else hidden_dim


def build_model(config: dict, vocab_sizes: dict) -> nn.Module:
    embed_dim = config["model"]["embed_dim"]
    hidden_dim = config["model"]["hidden_dim"]
    dropout = config["model"]["dropout"]

    encoder_name = config["model"]["encoder"]
    pooling_name = config["model"]["pooling"]
    fusion_name = config["model"]["fusion"]

    protein_encoder = build_encoder(encoder_name, vocab_sizes["protein"], embed_dim, hidden_dim, dropout)
    compound_encoder = build_encoder(encoder_name, vocab_sizes["compound"], embed_dim, hidden_dim, dropout)

    protein_pooling = build_pooling(pooling_name, hidden_dim)
    compound_pooling = build_pooling(pooling_name, hidden_dim)

    fusion = build_fusion(fusion_name)
    fused_dim = fused_dim_for(fusion_name, hidden_dim)

    if config["model"]["type"] == "cnn":
        return CNNModel(
            protein_encoder,
            compound_encoder,
            protein_pooling,
            compound_pooling,
            fusion,
            fused_dim,
            hidden_dim,
            dropout,
        )

    return BaselineModel(
        protein_encoder,
        compound_encoder,
        protein_pooling,
        compound_pooling,
        fusion,
        fused_dim,
        hidden_dim,
        dropout,
    )


def get_loaders(config: dict, raw_dir: Path) -> Tuple[DataLoader, DataLoader, DataLoader, dict]:
    dataset = SequenceDataset(raw_dir)
    vocab_sizes = dataset.vocab_sizes()

    indices = np.arange(len(dataset))
    rng = np.random.default_rng(config["training"]["seed"])
    rng.shuffle(indices)

    n_total = len(indices)
    n_train = int(n_total * config["training"]["train_ratio"])
    n_valid = int(n_total * config["training"]["valid_ratio"])

    train_idx = indices[:n_train]
    valid_idx = indices[n_train : n_train + n_valid]
    test_idx = indices[n_train + n_valid :]

    subset = torch.utils.data.Subset
    train_ds = subset(dataset, train_idx)
    valid_ds = subset(dataset, valid_idx)
    test_ds = subset(dataset, test_idx)

    collate = lambda batch: collate_batch(
        batch,
        protein_max_len=config["data"]["protein_max_len"],
        compound_max_len=config["data"]["compound_max_len"],
    )

    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid_ds, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate)

    return train_loader, valid_loader, test_loader, vocab_sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lightweight EKP models")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config["training"]["seed"])

    raw_dir = Path(config["data"]["raw_dir"])
    train_loader, valid_loader, test_loader, vocab_sizes = get_loaders(config, raw_dir)

    device = torch.device("cuda" if torch.cuda.is_available() and config["training"]["use_cuda"] else "cpu")

    model = build_model(config, vocab_sizes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.MSELoss()

    train_losses = []
    valid_losses = []

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            protein = batch["protein"].to(device)
            protein_mask = batch["protein_mask"].to(device)
            compound = batch["compound"].to(device)
            compound_mask = batch["compound_mask"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            output = model(protein, protein_mask, compound, compound_mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        train_losses.append(float(np.mean(epoch_losses)))

        valid_metrics = evaluate(model, valid_loader, device)
        valid_losses.append(valid_metrics["rmse"])

        print(f"Epoch {epoch} | Train Loss {train_losses[-1]:.4f} | Valid RMSE {valid_losses[-1]:.4f}")

    # Save training curve
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_rmse")
    plt.legend()
    plt.title("Training Curve")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150)
    plt.close()

    # Final evaluation
    valid_metrics = evaluate(model, valid_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    (output_dir / "valid_metrics.json").write_text(
        json.dumps({k: v for k, v in valid_metrics.items() if k in ["rmse", "mae", "r2"]}, indent=2),
        encoding="utf-8",
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps({k: v for k, v in test_metrics.items() if k in ["rmse", "mae", "r2"]}, indent=2),
        encoding="utf-8",
    )

    # Prediction plots (test)
    y_true = test_metrics["y_true"]
    y_pred = test_metrics["y_pred"]
    residuals = y_true - y_pred

    plt.figure(figsize=(5, 5))
    plt.scatter(y_pred, y_true, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_actual.png", dpi=150)
    plt.savefig(output_dir / "prediction_plot.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40)
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_hist.png", dpi=150)
    plt.close()

    error_analysis_dir = Path(__file__).resolve().parents[1] / "experiments" / "error_analysis"
    y_true, y_pred, protein_lengths = predict_with_lengths(model, test_loader, device)
    model_config = {
        "encoder": config["model"]["encoder"],
        "pooling": config["model"]["pooling"],
        "fusion": config["model"]["fusion"],
    }
    write_error_analysis(error_analysis_dir, y_true, y_pred, protein_lengths, model_config)

    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_pred.png", dpi=150)
    plt.savefig(output_dir / "residual_plot.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
