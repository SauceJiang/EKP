# Model Comparison

## Validation and Test Performance

| Model | Split | RMSE | MAE | R2 |
|-------|-------|------|-----|----|
| LinearRegression | valid | 1.6242 | 1.2329 | 0.0248 |
| LinearRegression | test | 1.6469 | 1.2692 | 0.0227 |
| Ridge | valid | 1.6242 | 1.2329 | 0.0248 |
| Ridge | test | 1.6469 | 1.2692 | 0.0227 |
| RandomForest | valid | 1.2609 | 0.8888 | 0.4123 |
| RandomForest | test | 1.3137 | 0.9285 | 0.3781 |
| XGBoost | valid | 1.3264 | 0.9824 | 0.3496 |
| XGBoost | test | 1.3881 | 1.0290 | 0.3057 |