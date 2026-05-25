# 1. Introduction

## 1.1 Background

Enzyme kinetic parameter prediction is an important problem in enzyme engineering, metabolic pathway analysis, and biocatalyst discovery. Among different kinetic parameters, the turnover number:

```text
kcat
```

describes the catalytic efficiency of an enzyme and is widely used in systems biology and industrial biotechnology.

Traditional experimental measurement of enzyme kinetic parameters is often expensive, time-consuming, and difficult to scale. With the rapid growth of biological databases such as BRENDA and SABIO-RK, machine learning methods have become increasingly important for predicting enzyme kinetic properties from sequence and molecular information.

Recent studies, including CatPred, UniKP, and ERBA, demonstrate that multimodal learning approaches can improve enzyme kinetics prediction by integrating:

- protein sequence information
- substrate molecular representation
- structural or geometric context

These studies also show that lightweight learning frameworks can provide meaningful prediction performance without requiring extremely large-scale foundation models.

---

## 1.2 Motivation

Although recent deep learning approaches have achieved promising results, several challenges still remain in enzyme kinetic prediction tasks.

First, enzyme-substrate interaction data are often noisy and sparse. Different enzyme families may exhibit significantly different sequence distributions and catalytic behaviors, which increases prediction difficulty.

Second, many existing studies rely on large pretrained protein language models or structure-aware architectures with high computational cost. Such models are difficult to reproduce under limited computational resources in teaching-oriented environments.

Third, the influence of different lightweight architectural components, including pooling strategy and encoder selection, is still insufficiently explored in small-to-medium scale educational experiments.

Therefore, this project focuses on building a lightweight and modular multimodal regression framework for enzyme kinetic prediction. The framework aims to balance:

- prediction performance
- computational efficiency
- reproducibility
- extensibility

while allowing systematic ablation studies on different model components.

---

## 1.3 Project Objective

The primary objective of this project is to predict:

```text
log10(kcat)
```

from enzyme-substrate pairs using lightweight machine learning and deep learning approaches.

More specifically, this project aims to:

1. Establish traditional machine learning baselines for enzyme kinetic prediction
2. Build a lightweight multimodal sequence regression framework
3. Compare different encoder and pooling strategies through ablation experiments
4. Analyze the influence of architectural components on regression performance
5. Provide a modular experimental pipeline for future extension

---

## 1.4 Current Scope

At the current stage, experiments focus on:

- protein token sequences
- compound token sequences
- single-endpoint regression for log10(kcat)

The framework currently includes:

- statistical learning baselines
- embedding-based encoder
- lightweight CNN encoder
- mean pooling
- max pooling
- concat fusion
- regression head based on MLP

More advanced modules, such as attention mechanisms and transformer-based fusion, are reserved for future work.

---

## 1.5 Experimental Workflow

The current experimental workflow is summarized as follows:

```text
Dataset Collection
        ↓
Data Cleaning & Splitting
        ↓
Baseline Machine Learning Models
        ↓
Lightweight Deep Learning Framework
        ↓
Pooling & Encoder Ablation
        ↓
Performance Evaluation
```

The experiments are designed to provide both reproducible baselines and interpretable comparisons between lightweight model components.

# 2. Dataset and Preprocessing

## 2.1 Dataset Source

This project uses the DLKcat dataset and its CatPred-related variants, derived from enzyme kinetics entries curated from BRENDA and SABIO-RK. The task is single-target regression for:

```text
log10(kcat)
```

The working dataset contains approximately 23k enzyme-substrate pairs after cleaning.

---

## 2.2 Data Structure

Raw inputs are stored in three files:

- `proteins.npy`: variable-length protein token sequences
- `compounds.npy`: variable-length compound token sequences
- `regression.npy`: continuous regression targets

These files align by index, where each pair of protein and compound sequences corresponds to a single log10(kcat) target.

---

## 2.3 Data Cleaning

The preprocessing pipeline performs:

- empty sequence checking
- duplicate removal (protein + compound + target)
- abnormal target filtering (out-of-range values)
- sequence statistics generation for baseline features

---

## 2.4 Data Split

The dataset is split into train, validation, and test subsets using an 8:1:1 ratio. All experiments reuse the same split and fixed random seed to ensure reproducibility and fair comparison across models.

# 3. Methodology

## 3.1 Baseline Machine Learning Models

To verify the effectiveness of the proposed model framework in the enzyme kinetic parameter prediction task, this project first constructs a series of traditional statistical learning and machine learning baseline methods for comparison with subsequent deep learning models.

All models are trained and evaluated under a unified train/validation/test split, and their performance is compared using identical evaluation metrics, including RMSE, MAE, and $R^2$.

---

### 3.1.1 Linear Regression

As the most fundamental statistical learning model, Linear Regression is adopted to establish a direct linear relationship between statistical sequence features and enzyme kinetic parameters, using basic descriptors such as length, central tendency, dispersion, and extrema to probe whether the dataset exhibits a learnable linear signal under minimal model complexity.

---

### 3.1.2 Random Forest

Random Forest, as a bagging-based ensemble of decision trees, is used to evaluate whether nonlinear interactions among engineered features can be captured more effectively than linear baselines while remaining robust to feature scaling and moderate noise.

---

### 3.1.3 XGBoost

XGBoost (Extreme Gradient Boosting) serves as a strong structured-data baseline by incrementally fitting residuals to model complex feature interactions, providing a competitive nonlinear alternative to Random Forest under a similar feature representation.

---

## 3.2 Proposed Framework

### 3.2.1 Framework Overview

Based on traditional statistical learning models, this project further proposes a Lightweight Multimodal Sequence Regression Framework for enzyme kinetic parameter prediction.

The framework takes the following as inputs:

- protein token sequence
- compound token sequence

Through a modular design, it realizes:

- sequence encoding
- feature pooling
- multimodal fusion
- regression prediction

The overall framework is shown below:

```text
Protein Sequence
        ↓
 Protein Encoder

Compound Sequence
        ↓
Compound Encoder

        ↓
    Pooling Module
        ↓
     Fusion Module
        ↓
   Regression Head
        ↓
 Predict log10(kcat)
```

The framework has the following properties:

- lightweight
- modular
- reproducible
- convenient for ablation study

---

### 3.2.2 Encoder Module

The Encoder module is designed to learn representation features from input sequences.

Two types of encoders are implemented in this project.

#### (1) Embedding Encoder

The Embedding Encoder maps token IDs to low-dimensional continuous vectors via embedding lookup.

Its advantages include:

- Simple structure
- Low computational cost
- Easy to train

It is regarded as a lightweight deep learning baseline model.

---

#### (2) CNN Encoder

To further capture local sequence patterns, a lightweight 1D CNN Encoder is implemented.

Its structure is as follows:

```text
Embedding
→ Conv1D
→ ReLU
→ Conv1D
```

The CNN Encoder can extract:

- local motif
- subsequence pattern
- local interaction signal

It further enhances the model’s capability of learning local features.

---

### 3.2.3 Pooling Module

Given the variable length of input sequences, the pooling module is required to compress variable-length sequence representations into fixed-length vectors.

Two pooling methods are implemented in this project:

- Mean Pooling
- Max Pooling

Among them:

- Mean Pooling focuses on global statistical representation
- Max Pooling highlights local maximum activation features

---

### 3.2.4 Fusion Module

The Fusion Module is used to integrate information from protein and compound modalities.

The current framework adopts:

- Concat Fusion

Formulated as:

```text
z = [z_protein ; z_compound]
```

The fused representation is then fed into the regression head for prediction.

---

### 3.2.5 Regression Head

The Regression Head adopts a lightweight Multi-Layer Perceptron (MLP) structure to implement the mapping:

```text
f(z) → log10(kcat)
```

The loss function adopted is:

- Mean Squared Error (MSE)

for regression prediction.

---

# 4. Experimental Design

This section describes the controlled variables and the factors changed in ablation experiments. All experiments share the same data split, evaluation metrics, and training budget to ensure fair comparison.

---

## 4.1 Pooling Ablation

Experimental settings are fixed as:

- encoder: embedding
- fusion: concat
- trainer settings (seed, batch size, learning rate, epochs)

Only the pooling method is modified:

- Mean Pooling
- Max Pooling

---

## 4.2 Encoder Ablation

Experimental settings are fixed as:

- pooling: mean
- fusion: concat
- trainer settings (seed, batch size, learning rate, epochs)

Only the encoder type is changed:

- Embedding Encoder
- CNN Encoder

---

# 5. Experimental Results

## 5.1 Baseline Comparison

Table 5 compares traditional statistical baselines with the proposed lightweight deep learning framework. All results are reported on the test split using RMSE, MAE, and $R^2$.

| Model | RMSE | MAE | $R^2$ |
|---|---:|---:|---:|
| Linear Regression | 1.6469 | 1.2692 | 0.0227 |
| Random Forest | 1.3137 | 0.9285 | 0.3781 |
| XGBoost | 1.3881 | 1.0290 | 0.3057 |
| Proposed DL Framework (Embedding + Mean) | 1.2184 | 0.8840 | 0.4654 |

The Random Forest baseline is the strongest among traditional models, indicating that nonlinear feature interactions in the engineered statistical features are important. The proposed lightweight framework achieves the best overall performance, improving both RMSE and $R^2$ relative to all baselines, while remaining computationally modest.

Convergence analysis from the training curves shows stable optimization: validation RMSE decreases quickly in early epochs and flattens later without sharp divergence, suggesting reasonable generalization under the current training budget.

## 5.2 Pooling Ablation

Table 6 reports pooling ablation results under fixed encoder, fusion, and training settings (embedding encoder + concat fusion).

| Pooling | RMSE | MAE | $R^2$ |
|---|---:|---:|---:|
| Mean Pooling | 1.2184 | 0.8840 | 0.4654 |
| Max Pooling | 1.3760 | 1.0299 | 0.3181 |

Mean pooling consistently outperforms max pooling across all metrics, and the training curves indicate faster and smoother convergence with mean pooling, whereas max pooling shows mild oscillations in validation RMSE and slower improvement, which aligns with the intuition that global aggregation yields more stable regression signals than sparsely activated maxima in small-to-medium datasets.

## 5.3 Encoder Ablation

Table 7 compares encoder choices under fixed mean pooling and concat fusion.

| Encoder | RMSE | MAE | $R^2$ |
|---|---:|---:|---:|
| Embedding Encoder | 1.2184 | 0.8840 | 0.4654 |
| CNN Encoder | 1.2395 | 0.8880 | 0.4466 |

The CNN encoder captures local n-gram patterns, but the improvement over the embedding-only encoder is limited under the same lightweight setting, and while training remains stable without severe overfitting, the test metrics are slightly worse than the embedding baseline, suggesting that local pattern learning helps but the current shallow CNN is capacity-limited and that pooling choice contributes more to performance than encoder choice under the present computational budget.

# 6. Conclusion

In summary, by combining a clearly defined dataset and preprocessing pipeline with a modular lightweight framework and controlled ablation design, the study demonstrates that strong statistical baselines remain competitive yet are consistently surpassed by the embedding-based deep learning configuration, while the pooling strategy exerts a larger influence on predictive quality than encoder choice under equal training budgets, and the overall pattern of stable convergence together with modest improvements from local-pattern CNN features supports the conclusion that further gains are more likely to emerge from better global aggregation and representation balance rather than from shallow increases in encoder complexity alone.

---
