# Error Analysis Summary (Test Set)

## Model Configuration
- encoder: embedding
- pooling: mean
- fusion: concat

## Figures
- **Prediction vs Ground Truth**: The scatter plot (pred_vs_true.png) compares predicted and true log10(kcat). The dashed diagonal denotes perfect agreement, so deviations quantify prediction error.
- **Residual Distribution**: The histogram (residual_distribution.png) shows the distribution of residuals $y_{true} - y_{pred}$, indicating bias and dispersion.
- **Absolute Error vs Protein Length**: The scatter plot (error_vs_protein_length.png) visualizes whether sequence length is associated with larger absolute errors.

## Summary Statistics

| Metric | Value |
|---|---:|
| Residual mean | 0.0168 |
| Residual std | 1.2183 |
| Max absolute error | 6.2826 |
| Pearson corr (pred vs true) | 0.6850 |

## Interpretation
The prediction vs ground truth plot demonstrates the overall calibration of the embedding + mean pooling + concat fusion model on the test set. A tight clustering around the diagonal indicates strong alignment, while wider spread highlights hard-to-predict samples. The residual histogram provides a complementary view of bias; a mean near zero suggests limited systematic over- or under-estimation, whereas heavy tails indicate occasional large errors.

The absolute error vs protein length plot is used to diagnose length-dependent failure modes. A rising envelope of errors for longer proteins would be consistent with information loss from truncation at the maximum sequence length and with higher compositional complexity in long sequences.

These observations are consistent with the pooling ablation results: mean pooling produced the most stable performance and the lowest overall error. From a representation standpoint, mean pooling preserves global signal and reduces variance relative to max pooling, which aligns with a residual distribution that is more symmetric and with fewer extreme outliers.

Potential causes of large errors include sequence truncation, noisy or heterogeneous assay conditions in the underlying dataset, and limited capacity to model long-range dependencies. These factors are more pronounced for long proteins and complex compound contexts, which may require richer encoders or length-aware pooling in future iterations.
