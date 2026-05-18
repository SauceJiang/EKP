# Experiment Comparison

| Model | Encoder | Pooling | Fusion | RMSE | MAE | R2 |
|---|---|---|---|---:|---:|---:|
| Linear Regression | N/A | N/A | N/A | 1.6469 | 1.2692 | 0.0227 |
| Random Forest | N/A | N/A | N/A | 1.3137 | 0.9285 | 0.3781 |
| XGBoost | N/A | N/A | N/A | 1.3881 | 1.0290 | 0.3057 |
| Embedding + Mean | embedding | mean | concat | 1.2184 | 0.8840 | 0.4654 |
| Embedding + Max | embedding | max | concat | 1.3760 | 1.0299 | 0.3181 |
| CNN + Mean | cnn | mean | concat | 1.2395 | 0.8880 | 0.4466 |

## Short Interpretation
- Linear Regression: linear baseline with limited capacity on nonlinear sequence features.
- Random Forest: strongest statistical baseline, capturing nonlinear interactions among engineered features.
- XGBoost: competitive but slightly weaker than Random Forest on this split.
- Embedding + Mean: best overall deep learning result with stable generalization.
- Embedding + Max: pooling choice hurts performance relative to mean pooling.
- CNN + Mean: local pattern modeling helps vs max pooling but does not surpass embedding + mean.

## Best-Performing Model
- Embedding + Mean pooling (test RMSE 1.2184, R2 0.4654).

## Observations on Pooling and Encoder Choice
- Mean pooling consistently outperforms max pooling for this dataset.
- The CNN encoder adds local motif modeling, but the gain is modest compared to the embedding baseline.
- Encoder changes matter less than pooling choice under the same training budget and split.
