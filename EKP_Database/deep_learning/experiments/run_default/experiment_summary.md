# Experiment Summary - run_default

## Model Configuration
- encoder: embedding
- pooling: mean
- fusion: concat
- regressor: MLP head
- embed_dim: 128
- hidden_dim: 128
- dropout: 0.1
- max_len: protein=512, compound=128

## Training Settings
- seed: 42
- batch_size: 64
- epochs: 10
- learning_rate: 0.001
- loss: MSE

## Final Metrics
- valid RMSE: 1.2579
- valid MAE: 0.9165
- valid R2: 0.4424
- test RMSE: 1.2184
- test MAE: 0.8840
- test R2: 0.4654

## Overfitting Observations
- Train loss decreases steadily.
- Valid RMSE decreases then flattens with a small bump around later epochs, suggesting mild overfitting but not severe.

## Training Stability Observations
- Curve is smooth and stable without divergence.
- Convergence is relatively fast (valid RMSE reaches ~1.3 by mid-epochs).

## Outputs
- training_curve.png
- pred_vs_actual.png
- residual_hist.png
- residual_vs_pred.png
- valid_metrics.json
- test_metrics.json
