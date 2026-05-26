# 日均精度 85% 模型改进方案

本文档给出当前 2025 年数据诊断、相关文献结论，以及把固定
`2025-12-01` 到 `2025-12-31` 测试集日均精度提升到 `85%` 的模型改进路线。

当前尚未达到 85%。已验证的最好结果仍是：

```text
mean_daily_accuracy = 0.7857284442214673
accuracy            = 0.7857284442214671
85%达标天数          = 9 / 31
```

项目主指标定义如下：

```text
单点相对误差 = abs(y_pred - y_true) / max(abs(y_true), 40)
日精度       = 1 - 当天96点单点相对误差均值
月日均精度   = 一个月日精度总和 / 当月评估天数
```

因此，当前平均相对误差是 `0.214272`；若要达到 `85%` 日均精度，平均相对误差
必须降到 `0.150000`，还需要降低约 `0.064272`。

## 当前验证结果

当前最佳结果文件：

```text
outputs/online_residual_best/reports/online_residual_best_summary.json
```

当前最佳预测文件：

```text
outputs/online_residual_best/predictions/online_residual_best_predictions.csv
```

该结果来自：

```text
state-aware GBM residual anchor
+ leakage-safe online residual calibration
```

在线残差校正只使用 2025-12 测试流中已经发生并已知的前序日期误差，不使用未来
真实价格，因此符合滚动日前预测的无泄漏约束。

## 2025 年数据诊断

2025-12 测试月明显比 2025-10/11 验证期更难，核心变化是高价频率显著上升。

| 时间窗口 | 行数 | 均价 | 标准差 | 40元地板价点数 | >=500高价点数 | 1000封顶点数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2025全年 | 35040 | 200.115 | 194.501 | 16552 (47.24%) | 1873 (5.35%) | 516 (1.47%) |
| 2025-10/11验证期 | 5856 | 184.954 | 175.050 | 2581 (44.07%) | 193 (3.30%) | 78 (1.33%) |
| 2025-12测试期 | 2976 | 233.883 | 242.495 | 1399 (47.01%) | 426 (14.31%) | 32 (1.08%) |

结论：

- 40 元地板价并不是当前主要矛盾，12 月地板价比例与全年接近；
- 12 月 `>=500` 高价点占比达到 `14.31%`，是验证期 `3.30%` 的四倍以上；
- 12 月价格分布的主要漂移来自中高价和高价状态，而不是低价状态。

## 当前误差来源

使用当前最佳预测文件运行：

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/14_analyze_accuracy_gap.py \
  --predictions outputs/online_residual_best/predictions/online_residual_best_predictions.csv \
  --top-days 12
```

得到的 2025-12 价格分段误差如下：

| 真实价格区间 | 点数 | 区间精度 | 相对误差占比 |
| --- | ---: | ---: | ---: |
| 40地板价 | 1399 | 0.938761 | 13.44% |
| 40-100 | 51 | 0.216785 | 6.26% |
| 100-200 | 111 | 0.309597 | 12.02% |
| 200-500 | 1083 | 0.751376 | 42.23% |
| 500-800 | 189 | 0.594760 | 12.01% |
| 800+ | 143 | 0.373632 | 14.05% |

这里最重要的发现是：

- 最大误差来源不是单纯的 `800+` 封顶附近，而是 `200-500` 中高价区；
- `500-800` 和 `800+` 的精度明显偏低，说明模型对高价边界和高价幅度都偏保守；
- 当前模型对 40 元地板价已经较好，继续主要优化地板价对总精度帮助有限；
- 最差小时集中在 `08, 07, 00, 17, 01, 21, 19, 18, 20, 06, 22, 05`，基本对应早晚及夜间供需紧张窗口。

因此，后续优化应优先解决：

```text
200-500中高价幅度预测
+ 500以上高价状态识别
+ 800以上高价幅度预测
```

而不是继续围绕 40 元地板价做小幅后处理。

## 现有特征的上限

当前项目已经用到的内部日前变量包括：

- `发电总出力预测`
- `竞价空间`
- `统一负荷预测`
- `抽蓄`
- `统一新能源预测`
- `联络线计划`
- 派生的 `净负荷`、`供需裕度`、`新能源占比`、`竞价空间占比`
- 价格滞后、价格滚动状态、日历、天气、天气误差历史、分位数代理特征

2025 年这些变量与电价的相关性如下：

| 特征 | 与价格相关性 | 与 high>=500 相关性 |
| --- | ---: | ---: |
| `竞价空间` | 0.5982 | 0.2392 |
| `净负荷` | 0.5979 | 0.2557 |
| `新能源占比` | -0.5909 | -0.2063 |
| `统一新能源预测` | -0.5868 | -0.2016 |
| `竞价空间占比` | 0.5845 | 0.2125 |
| `供需裕度` | 0.3325 | 0.1890 |

这些变量能解释价格的大方向，但对 `>=500` 高价状态的相关性只有 `0.2` 左右，
不足以稳定识别 2025-12 的高价漂移。

高价状态诊断也支持这个判断。使用当前全时段 anchor 预测和现有特征训练高价识别器：

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/13_diagnose_high_price_regime.py \
  --config config/residual_calibrator_weather_error_q2.yaml \
  --features data/processed/features.csv \
  --predictions outputs/residual_calibrated_train2024q2/predictions/gbm_residual_anchor_all_predictions.csv \
  --target-threshold 500 \
  --top-k 120
```

结果：

```text
valid positives: 193
test positives: 426
valid AUC: 0.945633
valid AP: 0.524903
test AUC: 0.847986
test AP: 0.444803
base test accuracy: 0.771104
top-120 lift-to-500 accuracy: 0.767776
top-120 hits: 68
```

解释：

- 现有特征在验证期表现很好，但迁移到 12 月后明显下降；
- top-120 只命中 68 个真实高价点，且简单抬升到 500 后总精度反而下降；
- 说明问题不是“模型不会调参”，而是当前特征对 12 月稀缺状态的可观测性不够。

外部稀缺性数据审计当前仍未通过：

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/15_audit_external_signal_file.py \
  --config config/external_scarcity_template.yaml
```

当前输出：

```text
external_signal_path: data/external/scarcity_signals.csv
status: missing
required_action: provide a CSV with Date or date+slot and scarcity columns
```

## 文献结论

结合项目中的 `2501.06180v1.pdf` 和外部电价预测文献，可以提炼出四个与当前任务直接相关的结论。

第一，电价预测不能只看历史价格。Weron 的电价预测综述强调，电价受负荷、新能源、燃料、市场规则、日历和系统状态共同影响，模型比较必须建立在一致数据和一致回测框架上。本项目固定 2025-12 测试集和统一 accuracy 指标，就是为了避免只在某个随机划分上看起来变好。

第二，外生变量是必要的。近年的日前、日内和平衡市场电价预测综述也指出，负荷、可再生能源、跨区联络、系统约束、市场规则和混合/集成模型是电价预测中的关键方向。当前项目已经使用 `统一负荷预测`、`统一新能源预测`、`联络线计划`、`竞价空间` 等字段，但缺少实际负荷、实际新能源、可用容量、备用裕度、机组检修或市场稀缺披露等更直接的稀缺性信号。

第三，仅使用基本面点预测不够。Uniejewski 和 Ziel 的 2025 年文章提出，把负荷、风电、光伏等基本面变量的概率预测或分位数预测加入电价模型，可以显著提高电价点预测精度。原因是电价由非线性的供给曲线和剩余负荷决定，`price(E[剩余负荷])` 通常不等于 `E[price(剩余负荷)]`。这与当前项目的现象一致：单一的 `统一负荷预测` 和 `统一新能源预测` 能给出方向，但不能稳定解释高价幅度。

第四，高价尖峰应作为单独状态建模。价格尖峰预测和两阶段电价预测文献通常会把正常价格和尖峰价格分开处理：先判断是否进入尖峰/稀缺状态，再预测该状态下的价格幅度。当前实验也证明，简单用统一回归器或统一残差校正，很容易把高价段压低。

## 推荐模型路线

推荐路线是：

```text
强 anchor + 外部稀缺性特征 + 概率基本面特征 + 稀缺状态分类器 + 分状态残差专家
```

### 第一层：保留当前最强 anchor

不要丢弃当前最佳模型。它已经把地板价和大多数普通价格处理到较好水平。

保留：

```text
state-aware GBM residual anchor
+ online residual calibration
```

后续模型只在 anchor 的短板区域做增量修正：

```text
final_pred = anchor_pred + regime_residual_adjustment
```

### 第二层：补充外部稀缺性数据

新增外部 CSV，不提交到 Git：

```text
data/external/scarcity_signals.csv
```

支持两种时间索引：

```text
Date, reserve_margin_forecast, available_capacity_forecast, ...
```

或：

```text
date, slot, reserve_margin_forecast, available_capacity_forecast, ...
```

最高优先级字段：

- `reserve_margin_forecast`：日前备用裕度预测；
- `available_capacity_forecast`：日前可用容量预测；
- `outage_capacity_declared`：已披露检修/停运容量；
- `market_scarcity_index`：市场稀缺性披露指标；
- `load_forecast_external`：外部负荷预测；
- `renewable_forecast_external`：外部新能源预测；
- `actual_load`：实际负荷；
- `actual_renewable`：实际新能源；
- `actual_generation`：实际发电；
- `actual_tie_line`：实际联络线；
- `realized_reserve_margin`：实际备用裕度。

泄漏规则：

- 日前已知的预测/披露字段可以直接用于目标时刻；
- 实际值只能生成前一日、前两日、前七日和滚动统计，不能使用目标日当前时刻真实值。

### 第三层：构造概率基本面特征

在有实际负荷、新能源、发电数据后，把目前的滚动分位数代理升级为文献中的概率基本面特征：

- 负荷预测误差的历史模拟分位数；
- 新能源预测误差的历史模拟分位数；
- 负荷、新能源、剩余负荷的分位数回归后处理；
- 剩余负荷 `q10/q50/q90/q95/q99`；
- 剩余负荷分位差，例如 `q90-q50`、`q95-q50`；
- 备用裕度分位数和低备用裕度概率。

目标不是只多加几个特征，而是让模型知道：

```text
同一个点预测下，基本面不确定性越大，高价风险越高。
```

### 第四层：稀缺状态分类器

新增一个无泄漏状态分类器，预测每个 15 分钟点属于哪种价格状态：

```text
floor40 / low / normal / middle_high_200_500 / high_500_800 / cap_800_plus
```

输入：

- anchor 预测值；
- floor/high/cap 概率；
- 备用裕度预测；
- 滞后实际备用裕度；
- 可用容量和检修容量；
- 剩余负荷分位数；
- 低新能源和高负荷压力指标；
- 小时、日期、节假日、工作日特征。

该分类器不直接覆盖价格，只输出状态概率，供后续残差专家使用。

### 第五层：分状态残差专家

针对不同价格区间训练不同残差专家：

- `middle_high_200_500`：当前最大误差来源；
- `high_500_800`：高价边界低估；
- `cap_800_plus`：封顶附近幅度低估；
- `low_40_200`：防止把低价和普通价格错误抬高。

建议第一版融合公式：

```text
final_pred = anchor_pred
           + P(mid_high) * residual_mid_high
           + P(high) * residual_high
           + P(cap) * residual_cap
           - P(false_high_lift) * conservative_down_adjustment
```

约束：

- 只有当稀缺状态概率超过验证集选择的阈值时才允许上调；
- 上调幅度按状态分位数截断，避免误报高价造成大误差；
- 总 `mean_daily_accuracy`、`200-500`、`500-800`、`800+` 必须同时改善。

### 第六层：锚定式深度残差模型

深度学习不要再从零预测完整价格曲线。前面的 GPU TCN 实验已经说明，直接全价格预测不如 GBM anchor 稳定。

只有在外部稀缺性数据通过诊断后，再训练锚定式深度残差模型：

```text
输入：过去14天序列 + 目标日已知预测 + 稀缺性/概率基本面特征
目标：true_price - anchor_price
输出：96点残差曲线 + 状态logits + 可选分位数头
损失：项目relative MAE + 中高价/高价加权残差损失 + 状态分类损失
```

采用原则：

```text
深度模型必须在验证期和2025-12测试期都超过 anchor，才允许替换当前 best。
```

## 实施步骤

### 1. 外部数据就绪检查

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/15_audit_external_signal_file.py \
  --config config/external_scarcity_template.yaml
```

通过条件：

```text
scarcity_modeling_readiness: ready_for_feature_merge
complete_96_slot_days: 31
核心字段缺失率 <= 5%
```

### 2. 合并外部稀缺性特征

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/12_add_external_signals.py \
  --config config/external_scarcity_template.yaml \
  --input-features outputs/prevday_curve_2025_priority/features_prevday_curve_weather_error.csv \
  --output-features outputs/external_scarcity/features_external_scarcity.csv
```

### 3. 先做高价识别诊断

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/13_diagnose_high_price_regime.py \
  --config config/residual_calibrator_weather_error_q2.yaml \
  --features outputs/external_scarcity/features_external_scarcity.csv \
  --predictions outputs/residual_calibrated_train2024q2/predictions/gbm_residual_anchor_all_predictions.csv \
  --target-threshold 500 \
  --top-k 120
```

建议通过门槛：

```text
test AP >= 0.60
top-120 hits >= 90
top-120 lift-to-500 accuracy > base test accuracy
```

如果这个门槛过不了，说明新增数据仍然不能稳定识别 12 月高价状态，此时直接训练更大模型也很难达到 85%。

### 4. 训练稀缺状态残差集成模型

新增：

```text
scripts/20_train_scarcity_regime_ensemble.py
config/scarcity_regime_ensemble.yaml
outputs/scarcity_regime_ensemble/
```

该脚本应完成：

- 状态分类器训练；
- 分状态残差专家训练；
- 验证集选择融合阈值和上调幅度；
- 固定 2025-12 测试集评估。

### 5. 再训练深度残差模型

只有步骤 3 通过后，再新增：

```text
config/deep_scarcity_residual.yaml
outputs/deep_scarcity_residual/
```

该模型作为高容量后续方案，不作为第一优先级。

## 85% 验收标准

只有同时满足以下条件，才认为实现了日均精度 85%：

- 固定测试窗口仍为 `2025-12-01` 到 `2025-12-31`；
- 不使用目标日当前时刻真实实际值，保证无泄漏；
- summary 中 `mean_daily_accuracy >= 0.850000`；
- 作为稳定性检查，`pass_days >= 16 / 31`；
- `200-500` 区间精度超过 `0.80`；
- `500-800` 区间精度超过 `0.70`；
- `800+` 区间精度超过 `0.55`；
- 结果可由提交的脚本和配置复现；
- 数据集和 `outputs/` 不进入 Git。

候选模型验证命令：

```bash
../dayahead_epf_agent_project/.venv/bin/python scripts/14_analyze_accuracy_gap.py \
  --predictions outputs/<candidate>/predictions/<candidate>_predictions.csv \
  --top-days 12
```

## 预期收益与风险

预期收益：

- 外部稀缺性信号应提高高价状态识别能力；
- 概率基本面特征应改善中高价和高价幅度预测；
- 分状态残差专家应重点降低 `200-500` 区间的最大误差占比；
- anchor 保留后，低价和普通价稳定性不会被深度模型大幅破坏。

主要风险：

- 如果没有 `data/external/scarcity_signals.csv`，或者缺少完整 96 点覆盖，当前内部特征路线大概率仍会停在 `0.78-0.79` 附近；
- 继续只调 GBM、阈值、在线残差、日偏置、analog-day 或普通 TCN，已有实验显示都没有突破当前 best；
- 因此，85% 的关键不是再做小调参，而是补上能解释 12 月供需稀缺状态的数据，并用状态化模型消化这些数据。

## 参考文献

- Weron, R. (2014). *Electricity price forecasting: A review of the
  state-of-the-art with a look into the future*. International Journal of
  Forecasting, 30(4), 1030-1081.
  https://doi.org/10.1016/j.ijforecast.2014.08.008
- Uniejewski, B. and Ziel, F. (2025). *Probabilistic Forecasts of Load, Solar
  and Wind for Electricity Price Forecasting*. arXiv:2501.06180.
  https://arxiv.org/abs/2501.06180
- *A Review of Electricity Price Forecasting Models in the Day-Ahead,
  Intra-Day, and Balancing Markets*. Energies, 18(12), 3097.
  https://www.mdpi.com/1996-1073/18/12/3097
- *Price Forecasting in the Day-Ahead Energy Market by an Iterative Method with
  Separate Normal Price and Price Spike Frameworks*. Energies, 6(11), 5897.
  https://www.mdpi.com/1996-1073/6/11/5897
