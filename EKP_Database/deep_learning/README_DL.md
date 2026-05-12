# Lightweight Deep Learning Pipeline

This folder contains a lightweight, modular PyTorch pipeline for EKP sequence regression.

## Structure
- datasets/: dataset and collate utilities
- encoders/: embedding, CNN, and MLP encoders
- pooling/: mean, max, attention pooling
- fusion/: concat, add, multiply
- models/: baseline and CNN models
- trainer/: training and evaluation utilities
- configs/: YAML config
- experiments/: run outputs and logs

## Quick Start
1. Configure settings in configs/default_config.yaml
2. Run training:

```bash
python -m EKP_Database.deep_learning.trainer.train
```

## Notes
- This pipeline is lightweight and avoids large models.
- Swap encoder, pooling, and fusion via config.
