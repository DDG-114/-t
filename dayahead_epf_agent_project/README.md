# Day-Ahead Electricity Price Forecasting Agent Project

本工程用于基于 `GS.csv` 数据集实现 **未来 24 小时、15 分钟粒度、每日 96 点** 的日前现货电价预测模型。

核心建议路线：

1. 先建立可解释、可交付的强 baseline；
2. 使用 **slot-wise LightGBM** 作为强基线；
3. 根据 report2 新增 **policy-aware TCN** 深度模型：历史 TCN 编码、未来协变量分支、政策边界条件化、直接输出 96 点；
4. 按日计算预测精度，目标为日平均精度 `>= 85%`；
5. 后续再扩展 XGBoost、LSTM/Transformer、预测区间和报价策略影响分析。

> 注意：数据中存在 `Price = 0` 的点，因此不能直接使用普通 MAPE。工程默认采用带最小分母的 modified relative error。

---

## 目录结构

```text
dayahead_epf_agent_project/
├── AGENT_TASK.md                 # 给 Coding Agent 的任务说明
├── README.md
├── requirements.txt
├── config/default.yaml
├── data/raw/GS.csv               # 原始数据，已从压缩包中复制并重命名
├── data/processed/               # 特征表输出
├── outputs/                      # 模型、预测、报告输出
├── src/epf/                      # 核心代码包
├── scripts/                      # 命令行入口
├── tests/                        # 简单单元测试
└── docs/                         # 设计说明
```

---

## 快速开始

```bash
cd dayahead_epf_agent_project
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

先检查数据：

```bash
python scripts/00_profile_data.py --config config/default.yaml
```

构建特征：

```bash
python scripts/01_build_features.py --config config/default.yaml
```

训练 baseline：

```bash
python scripts/02_train_baselines.py --config config/default.yaml
```

训练 slot-wise LightGBM：

```bash
python scripts/03_train_slot_lightgbm.py --config config/default.yaml
```

训练 report2 对齐的 policy-aware TCN：

```bash
python scripts/05_train_policy_tcn.py --config config/default.yaml
python scripts/06_evaluate_policy_tcn.py --config config/default.yaml
```

该模型需要正常安装 PyTorch。当前工程将 LightGBM 保留为强基线，TCN 作为新的深度学习主架构。

评估模型：

```bash
python scripts/04_evaluate.py --config config/default.yaml
```

一键运行：

```bash
python scripts/run_pipeline.py --config config/default.yaml
```

---

## 默认数据划分

当前配置采用 2025 年为主要训练与测试范围：

```text
Train: 2025-01-01 ~ 2025-09-30
Valid: 2025-10-01 ~ 2025-11-30
Test : 2025-12-01 ~ 2025-12-31
```

特征构建时可以使用 2024 年数据作为滞后特征来源，但训练目标日期按上述区间过滤。

---

## 核心模型

强基线模型：

```text
for slot in 0..95:
    train one LightGBMRegressor for that 15-minute delivery interval
```

新增深度模型：

```text
history window: 过去14天 × 96点
known future: 目标日96点负荷/新能源/竞价空间/联络线/抽蓄/日历/价格上下文
static policy: policy_regime + price_lower_bound + price_upper_bound

history -> causal dilated residual TCN
known future -> MLP/temporal encoder
policy/static -> gated conditioning
fused representation -> direct 96-step residual price head
optional sigma head -> heteroscedastic auxiliary loss
```

训练损失默认采用 report2 建议的业务对齐形式：

```text
weighted relative MAE + ramp regularization + 0.05 * heteroscedastic NLL
```

输出层不写死 2025 年 40 元下限，而是在预测后按年份/政策边界 clip：

```text
2025: [40, 1000]
2026: [0, 1000]
```

输出：

```text
D 日 96 个预测点：
P_hat[D, slot=0], P_hat[D, slot=1], ..., P_hat[D, slot=95]
```

---

## 评价指标

默认输出：

- MAE
- RMSE
- Modified MAPE
- Daily Accuracy = `1 - modified relative error mean`
- 日平均精度是否达到 `>= 85%`
- 每日 96 点预测结果

如果老师或甲方给出正式精度公式，应优先替换 `src/epf/metrics.py` 中的 `daily_accuracy` 实现。

工程同时输出一个辅助口径：

```text
Cap-normalized Accuracy = 1 - MAE / price_range
```

默认 `price_range=1000`，用于对照“价格上下限归一化”的 85% 口径。该口径不会替代上面的 modified relative error 主口径。

---

## 主要输出

一键运行后会生成：

```text
outputs/predictions/baseline_predictions.csv
outputs/predictions/slot_lightgbm_predictions.csv
outputs/predictions/future_24h_predictions.csv
outputs/reports/baseline_summary.json
outputs/reports/slot_lightgbm_summary.json
outputs/reports/slot_lightgbm_daily_metrics.csv
outputs/reports/future_24h_summary.json
outputs/models/lgbm_slot_00.pkl ... lgbm_slot_95.pkl
outputs/models/policy_aware_tcn.pt
outputs/models/policy_aware_tcn_report.json
outputs/predictions/policy_aware_tcn_predictions.csv
outputs/reports/policy_aware_tcn_summary.json
```

其中 `future_24h_predictions.csv` 是配置中 `forecast.target_date` 对应日期的 96 点预测结果。
