# 《数据挖掘与机器学习》个人 Big Project（2025-2026）

## 题目：PP 拉丝价格月度预测（问题1）

本仓库基于 `PP数据/` 下的原始 Excel（表0-16）完成问题1：预测 **PP拉丝价格的月度价格**，并输出 **未来涨跌方向**（严格按 `>0` 为涨，否则跌）。

### 我们已确定的关键口径（重要）

- **预测目标**：预测下月（t+1），在数据集中每一行代表“被预测的月份 t”
- **涨跌方向（问题1）**：`(y_t - y_{t-1}) > 0` 为涨，否则为跌（用于评估/输出方向）
- **涨跌强度（问题2）**：按涨跌幅分档（默认：`|r|<=0.5%` 记为`flat`，`|r|>=5%` 记为`big_*`；其余为`small_*`），并输出各档概率
- **降频聚合（周/日 → 月）**：
  - 若原表含 `最低/最高/平均`：
    - `mean` = 月内 `平均` 的均值
    - `min` = 月内 `最低` 的最小值
    - `max` = 月内 `最高` 的最大值
    - `last` = 月内最后一个观测日的 `平均`
    - `range` = `max - min`
  - 若原表不含 `最低/最高/平均`，但存在数值列（如 `inventory`、`值`）：统一把该数值列当作序列值，同样计算 `mean/min/max/last/range`
- **期货变量**：对 `PP数据/13-大商所PP期货价格_短数据.xlsx` 做“纳入/不纳入”两套方案对比

## 快速开始

### 1) 生成问题1的月度建模数据集

默认用“月均价（由日度`平均`聚合）”作为预测目标，可选 `--target-metric last` 改为“月末价”口径。

```bash
python scripts/build_q1_dataset.py --target-metric mean --output outputs/datasets/q1_long.csv
python scripts/build_q1_dataset.py --target-metric mean --include-futures --output outputs/datasets/q1_with_futures.csv
```

### 2) 训练/评估（时间序列连续窗口测试）

默认测试窗口：`2021-01` ~ `2021-07`（可用参数修改）。

```bash
python scripts/run_q1_models.py --dataset outputs/datasets/q1_long.csv
python scripts/run_q1_models.py --dataset outputs/datasets/q1_with_futures.csv --futures-mode restrict
```

输出会写入 `outputs/metrics/<dataset_stem>/`：

- `q1_model_metrics.csv`：价格回归 + 方向（并附带由回归派生的“强度分档”指标）
- `q1_test_predictions.csv`：逐月预测（含 `return_pred`、`strength_pred`）
- `q1_strength_model_metrics.csv`：强度多分类模型指标
- `q1_strength_test_predictions.csv`：强度多分类逐月预测与各档概率（`proba__*`）

## 绘图权限提示（如果你在本机/沙盒里遇到 Matplotlib 报错）

如果出现“默认路径不可写”的提示，建议在运行前设置：

```bash
export MPLCONFIGDIR=.mplconfig
```

## 目录结构（可复现）

- `PP数据/`：原始数据（不改动）
- `pp_forecast/`：可复用的数据处理/建模代码
- `scripts/`：可直接运行的脚本（生成数据集、训练评估）
- `notebooks/`：工作用 notebook（可选）
- `outputs/`：生成的数据集、图、指标结果
