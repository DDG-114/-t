# 数据处理后 LightGBM Baseline 搭建报告

## 1. 报告目的

本报告说明在完成数据处理后，如何将当前 slot-wise LightGBM 作为第一版可复现 baseline。这里的 baseline 不是简单的“昨日同点”复制法，而是一个经过数据清洗、特征工程、训练/验证/测试划分后的工程 baseline，用于后续模型对比。

当前 baseline 已采用相对误差加权 LightGBM。后续任何新模型，例如 XGBoost、两阶段模型、LSTM/Transformer 或集成模型，都应至少与本报告中的 LightGBM baseline 在同一测试集和同一指标口径下比较。

本报告中的“日均精度”和“月均日精度”默认采用相对误差口径：

```text
单点精度 = 1 - abs(预测电价 - 真实电价) / max(abs(真实电价), 40)
日均精度 = 当日有效点单点精度的平均值
月均日精度 = 当月每日精度的平均值
```

`1 - MAE / 1000` 仅作为辅助参考口径，不作为本报告的主达标口径。

## 2. Baseline 定义

当前 baseline 名称：

```text
relative-error weighted slot-wise LightGBM baseline
```

核心思想：

```text
一天有 96 个 15min slot。
每个 slot 单独训练一个 LightGBMRegressor。
最终把 96 个模型的输出拼接成未来 24 小时 96 点预测曲线。
```

模型文件：

```text
outputs/models/lgbm_slot_00.pkl
...
outputs/models/lgbm_slot_95.pkl
```

模型元数据：

```text
outputs/models/feature_columns.json
outputs/models/train_report.json
```

## 3. 输入数据与数据处理基础

原始数据：

```text
data/raw/GS.csv
```

处理后特征表：

```text
data/processed/features.csv
```

原始 CSV 规模：

```text
81504 行，8 列
```

有效时间戳解析后：

```text
81503 行
```

时间范围：

```text
2024-01-01 00:15:00 至 2026-04-28 23:45:00
```

数据粒度：

```text
15 分钟，每天理论上 96 个点
```

完整日情况：

```text
总天数：849
不完整日：2024-01-01，仅 95 个点
```

## 4. 价格标签清洗

当前工程使用清洗后的 `Price` 作为训练标签和评估标签，原始价格保留在 `Price_raw`。

清洗配置：

```yaml
target_cleaning:
  enabled: true
  cap_values_as_missing: [0.0, 1000.0]
  constant_run_min_length: 672
```

规则说明：

| 规则 | 处理 |
|---|---|
| `Price == 0` | 低于已知最低报价，视为异常占位标签，置为 `NaN` |
| `Price == 1000` | 视为异常封顶值，置为 `NaN` |
| 连续 672 个点及以上完全同价 | 视为超长常数异常段，置为 `NaN` |
| `Price == 40` | 作为已知最低报价，正常保留 |

由于 15 分钟一个点：

```text
672 点 = 7 天
```

因此当前不会删除短时间的正常地板价，只处理超长常数段。

清洗审计结果：

| 项目 | 数值 |
|---|---:|
| 总行数 | 81503 |
| 清洗行数 | 20182 |
| 清洗比例 | 24.76% |
| `long_constant_run` | 10009 |
| `cap_value` | 5975 |
| `cap_value;long_constant_run` | 4198 |

按年份：

| 年份 | 清洗行数 |
|---|---:|
| 2024 | 15620 |
| 2025 | 517 |
| 2026 | 4045 |

说明：

- 2025 年主要清洗 `1000`，共 516 个点；
- 2025 年仅 1 个点来自跨年超长 `40` 常数段尾部；
- 2024 与 2026 中的 `0` 已作为异常标签清洗；
- 正常的 `40` 地板价保留；
- `Price_raw` 不进入模型特征，避免标签泄露。

## 5. Baseline 特征体系

LightGBM baseline 使用 33 个数值特征，主要分为四类。

### 5.1 原始外生变量

| 特征 |
|---|
| `发电总出力预测` |
| `竞价空间` |
| `统一负荷预测` |
| `抽蓄` |
| `统一新能源预测` |
| `联络线计划` |

### 5.2 市场含义特征

| 特征 | 公式 |
|---|---|
| `净负荷` | `统一负荷预测 - 统一新能源预测` |
| `供需裕度` | `发电总出力预测 - 统一负荷预测` |
| `新能源占比` | `统一新能源预测 / 统一负荷预测` |
| `竞价空间占比` | `竞价空间 / 统一负荷预测` |
| `联络线占比` | `联络线计划 / 统一负荷预测` |

### 5.3 时间特征

| 特征 |
|---|
| `slot` |
| `hour` |
| `day_of_week` |
| `month` |
| `is_weekend` |
| `slot_sin` |
| `slot_cos` |
| `month_sin` |
| `month_cos` |

### 5.4 历史价格特征

历史价格特征基于清洗后的 `Price` 构建。

lag 特征：

| 特征 | 含义 |
|---|---|
| `price_lag_1d` | 前 1 天同 slot 价格 |
| `price_lag_2d` | 前 2 天同 slot 价格 |
| `price_lag_7d` | 前 7 天同 slot 价格 |
| `price_lag_14d` | 前 14 天同 slot 价格 |

rolling 特征：

| 特征 | 含义 |
|---|---|
| `price_roll_3d_mean` | 前 3 天同 slot 均值 |
| `price_roll_3d_median` | 前 3 天同 slot 中位数 |
| `price_roll_7d_mean` | 前 7 天同 slot 均值 |
| `price_roll_7d_median` | 前 7 天同 slot 中位数 |
| `price_roll_7d_std` | 前 7 天同 slot 标准差 |
| `price_roll_7d_max` | 前 7 天同 slot 最大值 |
| `price_roll_7d_min` | 前 7 天同 slot 最小值 |
| `price_roll_14d_mean` | 前 14 天同 slot 均值 |
| `price_roll_14d_median` | 前 14 天同 slot 中位数 |

## 6. 数据划分

当前 baseline 使用 2025 年作为主要训练和评估范围。

| 集合 | 日期范围 |
|---|---|
| Train | 2025-01-01 至 2025-09-30 |
| Valid | 2025-10-01 至 2025-11-30 |
| Test | 2025-12-01 至 2025-12-31 |

测试集规模：

```text
31 天 × 96 点 = 2976 行
```

清洗后有效评估点：

```text
2944 / 2976
```

被清洗或缺失的目标标签不参与监督评估。

## 7. 模型训练设置

模型训练入口：

```text
scripts/03_train_slot_lightgbm.py
```

核心训练模块：

```text
src/epf/train_slot_lgbm.py
```

当前 LightGBM 参数：

```yaml
objective: regression_l1
learning_rate: 0.03
n_estimators: 250
num_leaves: 31
max_depth: -1
min_child_samples: 10
subsample: 0.9
colsample_bytree: 0.9
reg_alpha: 0.1
reg_lambda: 0.5
random_state: 42
n_jobs: -1
verbosity: -1
```

当前 baseline 的训练目标已经从普通回归误差改为更贴近相对误差口径的加权 L1 回归。

训练样本权重：

```text
sample_weight = 1 / max(abs(Price), 40)
```

因此低价样本的误差会得到更高权重。例如：

```text
Price = 40  -> weight = 1/40
Price = 400 -> weight = 1/400
```

这不是直接优化“月均日精度”，但比普通回归更接近相对误差验收公式：

```text
abs(pred - true) / max(abs(true), 40)
```

## 8. 训练结果

当前共训练：

```text
96 个 slot 模型
```

训练状态：

```text
96 / 96 trained
```

特征数量：

```text
33
```

训练报告：

```text
outputs/models/train_report.json
```

## 9. 测试集指标

测试集：

```text
2025-12-01 至 2025-12-31
```

### 9.1 LightGBM baseline 总体指标

| 指标 | 数值 |
|---|---:|
| MAE | 121.80 |
| RMSE | 214.15 |
| 相对误差 MAPE | 0.4830 |
| 相对误差口径整体精度 | 51.70% |
| 相对误差口径截断后单点精度均值 | 63.86% |
| `/1000` 辅助归一化精度 | 87.82% |

说明：

- 相对误差口径为：`1 - abs(error) / max(abs(true), 40)`；
- `/1000` 归一化口径为：`1 - MAE / 1000`，仅作辅助参考；
- 当前主口径采用相对误差，因此该 baseline 仍未达到 85%；
- 相比未加权 LightGBM，相对误差口径整体精度从 -6.82% 提升至 51.70%；
- `/1000` 口径下的 87.82% 不能作为本报告的主达标结论。

### 9.2 月度指标

月度指标文件：

```text
outputs/reports/slot_lightgbm_monthly_metrics.csv
```

2025 年 12 月结果。这里的月均日精度按以下方式计算：

```text
先计算每天有效点的相对误差口径日均精度，
再对 2025-12 的 31 个日均精度取平均。
```

| 月份 | 天数 | 月均相对误差口径精度 | 月均截断后单点精度 | `/1000` 辅助月均精度 | 相对误差口径达标天数 |
|---|---:|---:|---:|---:|---:|
| 2025-12 | 31 | 51.88% | 63.93% | 87.87% | 0 |

### 9.3 每日指标

每日指标文件：

```text
outputs/reports/slot_lightgbm_daily_metrics.csv
```

当前测试集共有：

```text
31 个完整自然日
```

按主口径，即相对误差日均精度：

```text
0 / 31 天 >= 85%
```

按辅助 `/1000` 归一化口径：

```text
22 / 31 天 >= 85%
```

## 10. 与简单 baseline 对比

当前工程还输出了简单 baseline，用于证明 LightGBM baseline 的相对收益。

简单 baseline 文件：

```text
outputs/reports/baseline_summary.json
```

对比结果如下：

| 模型 | MAE | RMSE | 月均相对误差口径精度 | 相对误差口径达标天数 | `/1000` 辅助精度 |
|---|---:|---:|---:|---:|---:|
| 昨日同点 | 141.88 | 248.56 | -18.01% | 0 | 85.81% |
| 上周同点 | 155.55 | 248.51 | 2.29% | 0 | 84.44% |
| 近 7 日中位数 | 132.82 | 208.85 | -13.32% | 0 | 86.72% |
| 近 7 日均值 | 120.31 | 183.80 | -11.31% | 0 | 87.97% |
| 未加权 slot-wise LightGBM | 106.35 | 165.17 | -6.72% | 0 | 89.37% |
| 相对误差加权 slot-wise LightGBM | 121.80 | 214.15 | 51.88% | 0 | 87.82% |

结论：

```text
相对误差加权 LightGBM 明显改善了主口径，从 -6.72% 提升至 51.88%。
但在正式采用的相对误差口径下，当前 baseline 仍未达到 85%。
```

## 11. 未来 24 小时预测输出

当前未来预测目标日：

```text
2026-04-28
```

预测输出：

```text
outputs/predictions/future_24h_predictions.csv
```

预测摘要：

| 项目 | 数值 |
|---|---:|
| 目标日期 | 2026-04-28 |
| 输出点数 | 96 |
| 成功预测点数 | 96 |
| 缺失真实标签点数 | 95 |

该预测文件是 baseline 模型对未来 24 小时、15 分钟粒度电价曲线的输出。

## 12. 可复现命令

完整重跑 pipeline：

```bash
python3 scripts/run_pipeline.py --config config/default.yaml
```

仅重训 LightGBM baseline：

```bash
python3 scripts/03_train_slot_lightgbm.py --config config/default.yaml
```

仅评估并输出未来 24 小时预测：

```bash
python3 scripts/04_evaluate.py --config config/default.yaml
```

运行单元测试：

```bash
python3 -m pytest -q
```

当前测试结果：

```text
7 passed
```

## 13. Baseline 的定位与后续对比规则

本报告中的 LightGBM baseline 用于回答：

```text
在当前数据清洗和相对误差加权训练条件下，一个稳健、可复现的树模型能达到什么水平？
```

后续模型必须至少在以下方面与它比较：

1. 同一训练/验证/测试划分；
2. 同一目标清洗规则；
3. 同一特征可用性假设；
4. 同一日级和月级指标；
5. 同一未来 24 小时预测输出格式。

后续优先改进方向：

1. 继续做低价段校准或两阶段低价分类模型。
2. 增加外生变量缺失标记和补值策略。
3. 增加 XGBoost/global LightGBM 对照。
4. 增加按价区间和尖峰识别指标。
5. 扩展到正式 3 个月价格验收集。
