# 问题1：PP 价格月度预测与未来涨跌方向预判

对应 `2025年个人PJ.docx` 第 1 问：基于 `PP数据/` 的历史相关数据，对 **PP拉丝价格**做**月度预测**，并输出 **未来涨跌方向**（上涨/下跌）。

> 说明：更完整的“特征工程 + 描述性统计 + 可视化”请见：`特征工程与描述性统计分析.md`（脚本：`scripts/run_q1_eda.py`）。

---

## 1. 任务定义与口径（必须写清）

### 1.1 预测目标（y）

- **预测粒度**：月度（YYYY-MM）
- **预测口径**：支持两种口径
  - 月均价：月内日度 `平均` 的均值（默认）
  - 月末价：月内最后一个观测日的日度 `平均`
- **预测步长**：预测下月（t+1）
  - 数据集每一行对应“被预测的月份 t”
  - 特征统一使用月 t-1 的信息：对全体特征做 `shift(1)`，避免时间泄露

### 1.2 涨跌方向（direction）

严格按讨论口径：

- `direction_t = 1`（涨）当且仅当 `(y_t - y_{t-1}) > 0`
- 否则 `direction_t = 0`（跌/不涨）

---

## 2. 数据与预处理（概述）

### 2.1 数据来源

`PP数据/`（表 1-17，外加表 0 汇总）：

- 预测目标价格：`PP数据/1-华东市场PP粒市场价_法定工作日.xlsx`
- 上中下游因子：产量/进口/开工率/成本/库存/检修/期货/GDP 等（详见 `pp_forecast/q1_dataset.py:31`）

### 2.2 统一降频到月度

对日度/周度/不规则数据，统一聚合为月度特征：

- 若存在 `最低/最高/平均`：生成 `mean/min/max/last/range`
- 若是单一数值列（如库存、GDP 的 `值`）：同样生成 `mean/min/max/last/range`

### 2.3 缺失值处理（建模内完成）

建模 Pipeline 内对所有数值特征做：

- `median` 填补 + 缺失指示（MissingIndicator）
- 标准化（StandardScaler）

---

## 3. 实验设计（时间序列）

### 3.1 训练/测试划分

按题目提示使用连续时间窗作为测试集（避免随机打乱）：

- 测试窗口：`2021-01` ~ `2021-07`（可配置）
- 训练集：严格使用测试窗口之前的月份

### 3.2 评价指标

同时评估“价格点预测”和“方向预测”：

- 价格回归：MAE / RMSE / MAPE
- 涨跌方向：Accuracy / Precision / Recall

---

## 4. 建模方案与参数呈现

### 4.1 Baseline（必须有）

- Naive：`ŷ_t = y_{t-1}`
- Seasonal Naive：`ŷ_t = y_{t-12}`（覆盖不足时会自动跳过）

基线结果会写到：`outputs/metrics/<dataset_stem>/q1_baselines.json`。

一次运行示例（2021-01~2021-07 测试窗，基于 `q1_long`）：

|Baseline|MAE|RMSE|MAPE|方向 Accuracy|Precision|Recall|
|---|---:|---:|---:|---:|---:|---:|
|naive（ŷ=y_prev）|325.75|372.20|3.70%|0.57|0.00|0.00|
|seasonal_12（ŷ=y_{t-12}）|1336.20|1489.51|15.11%|0.57|0.00|0.00|

### 4.2 回归模型（预测价格 y）

实现位置：`pp_forecast/q1_models.py:103`

当前提供多种可对比模型（线性/非线性/集成），例如：

- 线性：Ridge / Lasso / ElasticNet / BayesianRidge / HuberRegressor
- 树模型：RandomForestRegressor / ExtraTreesRegressor / GradientBoostingRegressor / AdaBoostRegressor
- 距离/核方法：KNNRegressor / SVR(RBF)

每个模型的参数会写到：`outputs/metrics/<dataset_stem>/q1_params_<model>.json`（用于报告“参数设置情况”）。

### 4.3 可选：额外特征工程（提升表达能力）

启用 `--engineer-features` 后，会增加动量/滚动统计/价差比值/供需比等派生特征（见 `pp_forecast/feature_engineering.py`）。

---

## 5. 结果对比（一次运行示例）

以下为默认测试窗（2021-01~2021-07）的一次结果摘要；完整对比见 `outputs/metrics/<dataset_stem>/q1_model_metrics.csv`。

> 注意：`with_futures` 在 `restrict` 模式下会丢弃期货缺失月份，样本显著变短，指标不宜与 long sample 直接“公平”比较，更多用于“纳入期货是否增益”的敏感性分析。

|数据集|说明|最佳模型（按 RMSE）|MAE|RMSE|MAPE|方向 Accuracy|Precision|Recall|
|---|---|---:|---:|---:|---:|---:|---:|---:|
|`q1_long`|不含期货|`rf`|223.37|240.43|2.57%|0.86|0.75|1.00|
|`q1_long_engineered`|不含期货 + 特征工程|`ada`|183.89|215.15|2.09%|1.00|1.00|1.00|
|`q1_with_futures`|含期货（restrict）|`huber`|395.54|506.92|4.48%|0.57|0.50|0.67|
|`q1_with_futures_engineered`|含期货 + 特征工程（restrict）|`ada`|235.72|318.91|2.65%|0.86|1.00|0.67|

总体观察（就本次测试窗而言）：

- 在 long sample 上，额外特征工程能显著改善 RMSE/MAPE。
- 期货数据覆盖较短，`restrict` 后有效样本进一步变少；建议同时报告“不纳入期货”与“纳入期货”两套结果。

---

## 6. 可复现运行方式（建议写入报告附录/README）

```bash
# 1) 构建数据集（默认月均价口径）
conda run -n lchen python scripts/build_q1_dataset.py --target-metric mean --output outputs/datasets/q1_long.csv
conda run -n lchen python scripts/build_q1_dataset.py --target-metric mean --include-futures --output outputs/datasets/q1_with_futures.csv

# 可选：开启额外特征工程
conda run -n lchen python scripts/build_q1_dataset.py --engineer-features --output outputs/datasets/q1_long_engineered.csv
conda run -n lchen python scripts/build_q1_dataset.py --include-futures --engineer-features --output outputs/datasets/q1_with_futures_engineered.csv

# 2) 训练与评估（时间序列测试窗）
conda run -n lchen python scripts/run_q1_models.py --dataset outputs/datasets/q1_long.csv
conda run -n lchen python scripts/run_q1_models.py --dataset outputs/datasets/q1_with_futures.csv --futures-mode restrict
```
