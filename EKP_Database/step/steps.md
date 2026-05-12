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

## Next
- 如果需要提升性能：在不引入深度模型的前提下尝试更多统计学习方法或特征选择。
- 若后续扩展到 $K_m$ 或 $K_i$，按相同流程复用脚本。
