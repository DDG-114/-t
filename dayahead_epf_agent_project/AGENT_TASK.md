# Agent Implementation Task: Day-Ahead Electricity Price Forecasting

## 背景

本项目基于 `data/raw/GS.csv` 做电力现货市场日前电价预测。目标不是单纯做 AI 回归，而是建立一个可解释、可测试、可交付的电价预测工程原型，用于支撑“电力市场日前电价预测导则/标准建议案”的技术验证。

数据为 15 分钟粒度，主要字段：

- `Date`
- `Price`
- `发电总出力预测`
- `竞价空间`
- `统一负荷预测`
- `抽蓄`
- `统一新能源预测`
- `联络线计划`

预测目标：

- 预测未来 24 小时电价；
- 每 15 分钟一个点；
- 每天共 96 个点；
- 日平均预测精度目标：`>= 85%`。

## 核心要求

请在本工程基础上实现、补全并调试完整 pipeline：

1. 数据读取和清洗；
2. 特征工程；
3. baseline 模型；
4. slot-wise LightGBM 主模型；
5. 预测与评估；
6. 输出可复现实验结果；
7. 代码遵守 PEP8，函数和模块有清晰注释。

## 第一版主模型结构

使用 **slot-wise LightGBM**：

```text
一天有 96 个 slot。
对每个 slot 单独训练一个 LightGBMRegressor。
最后将 96 个模型输出拼接为一天 96 点预测曲线。
```

示例：

```text
slot 0  -> model_slot_00.pkl -> 预测 00:00
slot 1  -> model_slot_01.pkl -> 预测 00:15
...
slot 95 -> model_slot_95.pkl -> 预测 23:45
```

## 特征要求

至少包括：

### 历史价格特征

- `price_lag_1d`
- `price_lag_2d`
- `price_lag_7d`
- `price_lag_14d`
- `price_roll_3d_mean`
- `price_roll_7d_mean`
- `price_roll_7d_median`
- `price_roll_14d_mean`
- `price_roll_7d_std`
- `price_roll_7d_max`
- `price_roll_7d_min`

### 原始外生变量

- `发电总出力预测`
- `竞价空间`
- `统一负荷预测`
- `抽蓄`
- `统一新能源预测`
- `联络线计划`

### 电力市场含义特征

- `净负荷 = 统一负荷预测 - 统一新能源预测`
- `供需裕度 = 发电总出力预测 - 统一负荷预测`
- `新能源占比 = 统一新能源预测 / 统一负荷预测`
- `竞价空间占比 = 竞价空间 / 统一负荷预测`
- `联络线占比 = 联络线计划 / 统一负荷预测`

### 时间特征

- `slot`
- `hour`
- `day_of_week`
- `month`
- `is_weekend`
- `slot_sin`
- `slot_cos`

## 默认数据划分

优先使用 2025 年数据：

```text
Train: 2025-01-01 to 2025-09-30
Valid: 2025-10-01 to 2025-11-30
Test : 2025-12-01 to 2025-12-31
```

特征构建可以使用 2024 年数据作为历史 lag 来源，但模型训练目标日期按上述区间划分。

## 指标要求

必须输出：

1. 全测试集 MAE；
2. 全测试集 RMSE；
3. modified MAPE；
4. 每日 daily accuracy；
5. daily accuracy `>= 85%` 的天数比例；
6. 每天 96 点预测曲线 CSV；
7. summary JSON。

由于 `Price` 中存在 0，默认使用：

```text
relative_error = abs(y_pred - y_true) / max(abs(y_true), price_floor)
daily_accuracy = 1 - mean(relative_error over 96 slots)
```

默认 `price_floor = 40`，可在 `config/default.yaml` 中调整。

## 需要特别注意

1. 不要泄露未来数据。lag/rolling 特征只能使用预测日前已知的历史价格。
2. 对于预测日的外生变量，可以假设它们是日前已知预测值，例如负荷预测、新能源预测、竞价空间等。
3. 如果某日不足 96 点，评估日报告中应标记该日为 incomplete day。
4. `2026-04-28` 的 Price 大量缺失，不应作为监督测试标签。
5. 如果正式验收公式与当前 modified accuracy 不同，优先按正式公式修改。

## 实现完成后的验收命令

```bash
python scripts/run_pipeline.py --config config/default.yaml
pytest -q
```

## 后续增强方向

Agent 完成第一版后，可以继续做：

1. XGBoost 对照；
2. global LightGBM single model；
3. LSTM/Transformer seq2seq 对照；
4. model ensemble；
5. 低价/高价/尖峰电价分类模块；
6. conformal prediction 预测区间；
7. 预测误差对报价策略和系统运行影响分析。
