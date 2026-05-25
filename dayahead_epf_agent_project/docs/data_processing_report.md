# 日前电价预测数据处理报告

## 1. 报告目的

本报告说明当前日前电价预测工程中 `GS.csv` 的数据读取、审计、清洗、特征构建、数据划分和输出产物。目标是让数据处理过程可复现、可解释、可审计，并为后续模型训练、测试验收和部署交付提供依据。

当前工程目录：

```text
/home/kaga/Desktop/datang/dayahead_epf_agent_project
```

原始数据文件：

```text
data/raw/GS.csv
```

处理后特征文件：

```text
data/processed/features.csv
```

主要配置文件：

```text
config/default.yaml
```

## 2. 原始数据概况

原始数据字段如下：

| 字段 | 含义 |
|---|---|
| `Date` | 时间戳 |
| `Price` | 电价标签，即预测目标 |
| `发电总出力预测` | 外生变量 |
| `竞价空间` | 外生变量 |
| `统一负荷预测` | 外生变量 |
| `抽蓄` | 外生变量 |
| `统一新能源预测` | 外生变量 |
| `联络线计划` | 外生变量 |

原始 CSV 共有：

```text
81504 行，8 列
```

解析时间戳并删除无效时间戳后，进入后续处理的数据为：

```text
81503 行
```

时间范围：

```text
2024-01-01 00:15:00 至 2026-04-28 23:45:00
```

数据粒度：

```text
15 分钟
每天理论上 96 个点
```

完整日检查结果：

```text
总天数：849 天
不完整日：2024-01-01，仅 95 个点
```

## 3. 数据读取与基础规范化

数据读取逻辑位于：

```text
src/epf/data.py
```

处理步骤如下：

1. 使用 `utf-8-sig` 编码读取 CSV。
2. 将 `Date` 转换为 `datetime`。
3. 删除 `Date` 无法解析的行。
4. 将 `Price` 和全部外生变量转换为数值型。
5. 按 `Date` 升序排序。
6. 删除重复时间戳。
7. 执行目标电价清洗。

当前配置：

```yaml
data:
  encoding: utf-8-sig
  expected_slots_per_day: 96
  slot_minutes: 15
```

## 4. 目标电价清洗规则

目标电价清洗的对象是 `Price`。清洗规则配置在：

```yaml
data:
  target_cleaning:
    enabled: true
    cap_values_as_missing: [0.0, 1000.0]
    constant_run_min_length: 672
```

### 4.1 保留原始标签

清洗不会覆盖原始信息。工程会新增两个字段：

| 字段 | 说明 |
|---|---|
| `Price_raw` | 清洗前的原始电价 |
| `Price_clean_reason` | 清洗原因 |

清洗后的 `Price` 用于：

- lag 特征；
- rolling 特征；
- 监督训练；
- 测试集评估。

`Price_raw` 只用于审计，不允许进入模型特征，避免标签泄露。

### 4.2 `Price == 0` 与 `Price == 1000` 的处理

当前业务判断中，`40` 是最低报价，因此 `0` 不作为有效成交电价；`1000` 被视为异常值或封顶占位值。因此：

```text
Price == 0    -> Price 置为 NaN
Price == 1000 -> Price 置为 NaN
Price_clean_reason = cap_value
```

这类数据不进入监督训练，也不进入正式评估。

### 4.3 连续常数段处理

业务上 `40` 是最低报价，因此短时间或正常市场条件下的 `40` 不应被删除。

为避免误删正常地板价，当前只清洗超长连续常数段：

```text
constant_run_min_length = 672
```

由于数据是 15 分钟粒度：

```text
672 点 = 7 天
```

因此规则为：

```text
同一电价连续 7 天及以上不变 -> Price 置为 NaN
Price_clean_reason = long_constant_run
```

这个规则主要用于处理明显不符合真实市场波动的超长常数段。

### 4.4 当前清洗结果

清洗审计文件：

```text
outputs/reports/price_cleaning_summary.json
outputs/reports/price_cleaning_rows.csv
```

当前结果：

| 项目 | 数值 |
|---|---:|
| 总行数 | 81503 |
| 清洗行数 | 20182 |
| 清洗比例 | 24.76% |
| `long_constant_run` | 10009 |
| `cap_value` | 5975 |
| `cap_value;long_constant_run` | 4198 |

按年份统计：

| 年份 | 清洗行数 |
|---|---:|
| 2024 | 15620 |
| 2025 | 517 |
| 2026 | 4045 |

按年份和原因统计：

| 年份 | 原因 | 行数 |
|---|---|---:|
| 2024 | `cap_value` | 1414 |
| 2024 | `cap_value;long_constant_run` | 4198 |
| 2024 | `long_constant_run` | 10008 |
| 2025 | `cap_value` | 516 |
| 2025 | `long_constant_run` | 1 |
| 2026 | `cap_value` | 4045 |

被清洗的原始价格主要包括：

| 原始价格 | 清洗行数 |
|---|---:|
| 0.0 | 9440 |
| 400.0 | 7105 |
| 40.0 | 2904 |
| 1000.0 | 733 |

说明：

- `0` 与 `1000` 均按异常标签清洗；
- 正常 `40` 地板价保留；
- 只有超长连续 `40` 常数段被清洗；
- 2025 年主要清洗 `1000`，另有 1 个点属于跨年超长 `40` 常数段尾部。

## 5. 缺失值情况

清洗后主要缺失情况如下：

| 字段 | 缺失数 |
|---|---:|
| `Price` | 20277 |
| `发电总出力预测` | 96 |
| `抽蓄` | 95 |
| `Price_raw` | 95 |

解释：

- `Price` 缺失包含原始目标缺失与清洗造成的缺失；
- `Price_raw` 缺失为原始数据中本身缺少价格的行；
- `2026-04-28` 有 95 个 `Price` 缺失，用作未来 24 小时预测目标日，不作为监督评估标签；
- 外生变量中的 `抽蓄` 在未来目标日存在缺失，LightGBM 可以原生处理 `NaN`。

## 6. 时间特征构建

时间特征由 `Date` 派生：

| 字段 | 说明 |
|---|---|
| `date` | 日期 |
| `hour` | 小时 |
| `minute` | 分钟 |
| `slot` | 15 分钟槽位，0 至 95 |
| `day_of_week` | 星期几 |
| `month` | 月份 |
| `is_weekend` | 是否周末 |
| `slot_sin` | 槽位周期正弦编码 |
| `slot_cos` | 槽位周期余弦编码 |
| `month_sin` | 月份周期正弦编码 |
| `month_cos` | 月份周期余弦编码 |

其中：

```text
slot = (hour * 60 + minute) / 15
```

## 7. 市场含义特征构建

根据原始外生变量构建以下市场含义特征：

| 特征 | 公式 |
|---|---|
| `净负荷` | `统一负荷预测 - 统一新能源预测` |
| `供需裕度` | `发电总出力预测 - 统一负荷预测` |
| `新能源占比` | `统一新能源预测 / 统一负荷预测` |
| `竞价空间占比` | `竞价空间 / 统一负荷预测` |
| `联络线占比` | `联络线计划 / 统一负荷预测` |

除法计算使用安全除法，分母为 0 时返回缺失值。

## 8. 历史价格特征构建

历史价格特征使用清洗后的 `Price` 构建，避免异常标签进入 lag 和 rolling 特征。

当前 lag 特征：

| 特征 | 含义 |
|---|---|
| `price_lag_1d` | 前 1 天同 slot 电价 |
| `price_lag_2d` | 前 2 天同 slot 电价 |
| `price_lag_7d` | 前 7 天同 slot 电价 |
| `price_lag_14d` | 前 14 天同 slot 电价 |

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

rolling 特征只使用当前预测点之前的历史价格，避免未来信息泄露。

## 9. 模型特征列排除规则

模型输入特征由 `src/epf/features.py` 自动筛选数值列，但以下列明确排除：

| 排除列 | 原因 |
|---|---|
| `Price` | 预测目标 |
| `Price_raw` | 原始标签，避免泄露 |
| `Price_clean_reason` | 审计字段，不作为数值特征 |
| `Date` | 原始时间戳 |
| `date` | 日期对象 |
| `minute` | 已由 slot/hour 表达 |

当前模型特征数：

```text
33 个
```

## 10. 数据划分

当前默认划分使用 2025 年数据：

| 集合 | 日期范围 |
|---|---|
| Train | 2025-01-01 至 2025-09-30 |
| Valid | 2025-10-01 至 2025-11-30 |
| Test | 2025-12-01 至 2025-12-31 |

测试集：

```text
31 天 × 96 点 = 2976 行
```

清洗后有效评估点：

```text
2944 / 2976
```

无标签或被清洗为异常标签的点不参与监督评估。

## 11. 未来 24 小时预测目标日

当前配置：

```yaml
forecast:
  target_date: '2026-04-28'
  output_prefix: future_24h
```

该日共有 96 个 15 分钟点，其中 95 个 `Price` 缺失，因此作为未来 24 小时预测目标日。

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

## 12. 当前数据处理产物

主要产物如下：

| 文件 | 内容 |
|---|---|
| `data/processed/features.csv` | 清洗后特征表 |
| `outputs/reports/data_profile.json` | 数据概况 |
| `outputs/reports/price_cleaning_summary.json` | 价格清洗汇总 |
| `outputs/reports/price_cleaning_rows.csv` | 被清洗的具体行 |
| `outputs/predictions/baseline_predictions.csv` | baseline 测试集预测 |
| `outputs/predictions/slot_lightgbm_predictions.csv` | LightGBM 测试集预测 |
| `outputs/predictions/future_24h_predictions.csv` | 未来 24 小时 96 点预测 |
| `outputs/reports/slot_lightgbm_daily_metrics.csv` | LightGBM 每日指标 |
| `outputs/reports/slot_lightgbm_monthly_metrics.csv` | LightGBM 月度指标 |
| `outputs/reports/slot_lightgbm_summary.json` | LightGBM 汇总指标 |

## 13. 可复现命令

检查价格清洗：

```bash
python3 scripts/check_price_cleaning.py --config config/default.yaml
```

构建特征：

```bash
python3 scripts/01_build_features.py --config config/default.yaml
```

完整 pipeline：

```bash
python3 scripts/run_pipeline.py --config config/default.yaml
```

运行测试：

```bash
python3 -m pytest -q
```

当前测试结果：

```text
7 passed
```

## 14. 当前处理策略的边界与后续建议

当前版本已经实现：

- 时间戳解析；
- 15 分钟 slot 构建；
- 目标电价异常清洗；
- 原始标签审计保留；
- 市场含义特征；
- 历史 lag 与 rolling 特征；
- 训练、验证、测试切分；
- 未来 24 小时预测日识别；
- 清洗报告与指标报告输出。

仍建议后续补充：

1. 对外生变量增加缺失补值策略和缺失标记，例如 `*_missing_flag`、`*_imputed_flag`。
2. 对明显数量级异常的外生变量增加 Hampel/MAD 或业务边界清洗。
3. 继续优化相对误差加权 LightGBM，使训练目标更接近日均相对精度。
4. 扩展正式三个月验收集，并按自然月输出月均日精度。
