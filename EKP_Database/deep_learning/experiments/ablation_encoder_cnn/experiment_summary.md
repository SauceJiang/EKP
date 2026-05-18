# Experiment Summary - ablation_encoder_cnn

## Model Configuration
- encoder: cnn (Embedding -> Conv1D -> ReLU -> Conv1D)
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
- valid RMSE: 1.2526
- valid MAE: 0.9006
- valid R2: 0.4471
- test RMSE: 1.2395
- test MAE: 0.8880
- test R2: 0.4466

## CNN Representation Analysis
- The two-layer Conv1D stack emphasizes local n-gram patterns in both protein and compound sequences.
- Mean pooling over CNN features keeps the representation stable while preserving local motif cues.

## Performance Comparison
- CNN improves on max pooling + embedding but is slightly worse than embedding + mean pooling.
- Validation and test metrics remain close to the embedding baseline, indicating modest gains in local pattern capture without a large performance jump.

## Training Stability Observations
- Training remains stable with smooth RMSE evolution and no divergence.
- No strong overfitting is observed; validation RMSE tracks training loss without late-epoch spikes.

## Outputs
- training_curve.png
- prediction_plot.png
- residual_plot.png
- pred_vs_actual.png
- residual_hist.png
- residual_vs_pred.png
- valid_metrics.json
- test_metrics.json
