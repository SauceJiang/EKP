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
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40)
    plt.title("Residual Histogram")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_dir / "residual_vs_pred.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
