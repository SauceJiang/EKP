# DLKcat 数据处理记录

## Step 1: 确认 proteins/compounds 编码形态
- 已确认 `proteins.npy` 和 `compounds.npy` 为变长整数序列（token id）。
- 统计得到：
  - proteins token 范围 0~8584，长度区间 7~2515（均值 430.46）。
  - compounds token 范围 8~5427，长度区间 1~1096（均值 44.18）。
- 当前阶段不进行词表语义重建，专注统计特征与基线模型。

## Step 2: 建立清洗脚本并执行
- 在 `scripts/clean.py` 中实现了：
  - 读取原始 `.npy`。
  - 空值与空序列检查。
  - 重复样本剔除（按 protein+compound+target）。
  - 目标值异常范围检查（$<-6$ 或 $>6$ 视为异常并剔除）。
- 脚本已执行，输出：
  - `interim/cleaned/dlkcat_cleaned.csv`

## Step 3: 记录字段字典
- 在 `metadata/data_dictionary.xlsx` 中新增 `dlkcat_cleaned` 与 `dlkcat_statistical_features` sheet。
- 说明 token 字段为整数 id 序列，不做语义重建。

## Step 4: 统计特征工程与切分
- 生成统计特征表：`interim/statistical_features/dlkcat_statistical_features.csv`。
- 按 8/1/1 切分：
  - `processed/train/dlkcat_train.csv`
  - `processed/valid/dlkcat_valid.csv`
  - `processed/test/dlkcat_test.csv`

## Step 5: EDA 与基线模型
- EDA 结果输出至 `results/eda/`：
  - 目标分布、特征直方图与箱线图
  - 相关性矩阵与热力图
  - `eda_summary.md` 与 `correlation_analysis.csv`
- 基线模型结果输出至 `results/models/`：
  - `baseline_results.csv`
  - `model_comparison.md`
  - `feature_importance.csv`
  - `residual_analysis.md`
  - `prediction_plots/` 下的可视化图

  ## Step 6: 轻量化深度学习框架
  - 新增 `deep_learning/` 目录与模块化结构：
    - datasets: Dataset 与 collate
    - encoders: embedding/cnn/mlp
    - pooling: mean/max/attention
    - fusion: concat/add/multiply
    - models: baseline/cnn
    - trainer: train/evaluate/metrics
    - configs: 默认配置
    - experiments: 实验记录
  - 目标：支持替换编码器、池化与融合方式，便于后续消融。

## Next
- 在不引入大型模型的前提下，尝试轻量化深度学习基线与消融。
- 若后续扩展到 $K_m$ 或 $K_i$，按相同流程复用脚本。

## Step 7: CNN Encoder Ablation (Mean Pooling + Concat)
- Implemented a lightweight 1D CNN encoder (Embedding -> Conv1D -> ReLU -> Conv1D) for both protein and compound branches.
- Kept pooling, fusion, data split, and trainer settings fixed to isolate encoder effects.
- Result: CNN + mean pooling achieved test RMSE 1.2395 and R2 0.4466, improving over embedding + max pooling but slightly below embedding + mean pooling.

## Step 8: Formal Experiment Comparison Table
- Consolidated statistical baselines (Linear Regression, Random Forest, XGBoost) and deep learning runs (Embedding+Mean, Embedding+Max, CNN+Mean).
- Produced a paper-ready CSV/Markdown table with RMSE/MAE/R2 and short interpretations.
- Conclusion: mean pooling is consistently stronger than max pooling; encoder changes are secondary to pooling choice under the same budget.

## Step 9: Framework Diagram
- Created a clean architecture diagram for the lightweight multimodal sequence regression framework.
- Diagram highlights encoder, pooling, fusion, and regression head modules for reproducibility-focused reporting.

## Step 10: Experimental Results Section
- Reviewed baseline outputs in `results/models/` and DL experiment summaries in `deep_learning/experiments/`.
- Compiled RMSE/MAE/R2 comparisons for baseline models, pooling ablation (mean vs max), and encoder ablation (embedding vs CNN).
- Observed that mean pooling provides faster, more stable convergence and better test performance; encoder changes yield smaller gains under the current lightweight setting.

## Step 11: Paper Structure Refactor
- Refactored the paper layout to separate methodology, experimental design, and experimental results for clearer scientific reporting.
- This separation prevents overlap between setup descriptions and performance discussion, improving reproducibility and reader traceability.
- Added a dedicated dataset and preprocessing section covering data sources, file structure, cleaning, and the 8:1:1 split.
