"""Train baseline regression models and write evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results" / "models"
PLOTS_DIR = RESULTS_DIR / "prediction_plots"

TARGET_COL = "log10_kcat"
RANDOM_SEED = 42


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED_DIR / name / f"dlkcat_{name}.csv")
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return x, y


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    x_train, y_train = load_split("train")
    x_valid, y_valid = load_split("valid")
    x_test, y_test = load_split("test")

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=RANDOM_SEED),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
        ),
    }

    results = []
    predictions = {}

    for name, model in models.items():
        model.fit(x_train, y_train)

        pred_valid = model.predict(x_valid)
        pred_test = model.predict(x_test)

        valid_metrics = evaluate(y_valid, pred_valid)
        test_metrics = evaluate(y_test, pred_test)

        results.append({"model": name, "split": "valid", **valid_metrics})
        results.append({"model": name, "split": "test", **test_metrics})

        predictions[name] = {
            "valid": pred_valid,
            "test": pred_test,
            "model": model,
        }

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "baseline_results.csv", index=False)

    # Model comparison
    compare_lines = [
        "# Model Comparison",
        "",
        "## Validation and Test Performance",
        "",
        "| Model | Split | RMSE | MAE | R2 |",
        "|-------|-------|------|-----|----|",
    ]
    for _, row in results_df.iterrows():
        compare_lines.append(
            f"| {row['model']} | {row['split']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
        )

    (RESULTS_DIR / "model_comparison.md").write_text("\n".join(compare_lines), encoding="utf-8")

    # Feature importance
    feature_importance_rows = []
    for name in ["RandomForest", "XGBoost"]:
        model = predictions[name]["model"]
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, val in zip(x_train.columns, importances):
                feature_importance_rows.append({"model": name, "feature": feat, "importance": float(val)})

            top = pd.DataFrame({"feature": x_train.columns, "importance": importances}).sort_values(
                "importance", ascending=False
            )
            plt.figure(figsize=(7, 5))
            plt.barh(top["feature"].head(20)[::-1], top["importance"].head(20)[::-1])
            plt.title(f"Top 20 Feature Importances: {name}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"feature_importance_{name}.png", dpi=150)
            plt.close()

    pd.DataFrame(feature_importance_rows).to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

    # Residual analysis for best model on validation RMSE
    valid_rmse = results_df[results_df["split"] == "valid"].set_index("model")["rmse"]
    best_model_name = valid_rmse.idxmin()
    best_model = predictions[best_model_name]["model"]

    test_pred = predictions[best_model_name]["test"]
    residuals = y_test.values - test_pred

    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=40)
    plt.title(f"Residual Histogram: {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(test_pred, y_test, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Predicted vs Actual: {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "pred_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.scatter(test_pred, residuals, alpha=0.5)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title(f"Residual vs Predicted: {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "residual_vs_pred.png", dpi=150)
    plt.close()

    residual_lines = [
        "# Residual Analysis",
        "",
        f"Best model based on validation RMSE: {best_model_name}",
        "",
        "Plots:",
        "- residual_hist.png",
        "- pred_vs_actual.png",
        "- residual_vs_pred.png",
    ]
    (RESULTS_DIR / "residual_analysis.md").write_text("\n".join(residual_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
