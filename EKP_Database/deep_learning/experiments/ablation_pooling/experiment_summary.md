# Experiment Summary - ablation_pooling (max pooling)

## Model Configuration
- encoder: embedding
- pooling: max
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
- valid RMSE: 1.4421
- valid MAE: 1.0848
- valid R2: 0.2671
- test RMSE: 1.3760
- test MAE: 1.0299
- test R2: 0.3181

## Overfitting Observations
- Train loss decreases steadily while valid RMSE improves slowly.
- A small oscillation appears mid-training, indicating mild instability compared to mean pooling.

## Training Stability Observations
- Convergence is slower than the default mean pooling run.
- Validation RMSE remains higher throughout training.

## Comparison vs run_default
- Validation RMSE is worse (1.4421 vs 1.2579).
- Training stability is lower (more oscillation in valid RMSE).
- Convergence speed is slower (valid RMSE stays ~1.5+ for longer).

## Outputs
- training_curve.png
- pred_vs_actual.png
- residual_hist.png
- residual_vs_pred.png
- valid_metrics.json
- test_metrics.json
