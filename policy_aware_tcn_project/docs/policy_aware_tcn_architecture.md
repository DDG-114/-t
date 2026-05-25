# Policy-Aware TCN Architecture

This is the standalone deep model architecture added from `deep-research-report2.md`. It is designed as the first production-grade neural route. The LightGBM baseline is intentionally kept outside this project, in `../dayahead_epf_agent_project`, so the two implementations can be compared without mixing code paths.

## Contract

The model predicts one target day as a complete 96-point vector. It uses three input groups:

| Input | Shape | Meaning |
|---|---:|---|
| `x_hist` | `B x 1344 x F_hist` | Previous 14 complete calendar days, including historical price, exogenous forecasts, market-derived fields, time encodings, and price lag/rolling context |
| `x_fut` | `B x 96 x F_fut` | Target-day known-future covariates and same-slot historical context |
| `x_static` | `B x 4` | `policy_regime`, `price_lower_bound`, `price_upper_bound`, `is_zero_floor_regime` |
| `anchor` | `B x 96` | Yesterday's same-slot price anchor for residual prediction |

The implementation tolerates missing target labels by carrying a `y_mask`. Missing labels do not participate in loss or metrics, but missing inputs remain visible through appended missing-indicator channels.

## Architecture

The model is implemented in `src/epf/deep_models.py` as `PolicyAwareTCN`.

```text
x_hist
  -> 1x1 projection
  -> causal dilated residual TCN blocks, dilation 1/2/4/8/16/32
  -> final history context

x_fut
  -> known-future temporal encoder

x_static
  -> policy/static encoder

history context + future encoding + static encoding
  -> gated fusion per delivery slot
  -> residual price head
  -> anchor + bounded residual
  -> mu[96]

optional:
  -> log_sigma[96] for heteroscedastic auxiliary NLL
```

Default size is intentionally compact: 64 channels, six TCN blocks, dropout 0.15, and direct 96-step output. This matches the report's recommendation to avoid oversized Transformer-style models under 2025-only supervision.

## Loss

Training uses:

```text
weighted_relative_mae + 0.10 * ramp_consistency + 0.05 * heteroscedastic_nll
```

The relative denominator is `max(abs(y), 40)`, aligned with the configured daily accuracy formula. Point weights emphasize:

- high-price slots;
- low-price slots;
- near-boundary floor/cap slots.

Ramp consistency compares adjacent 15-minute price changes, so the model is penalized for unrealistic day-ahead curve shape even when pointwise error is acceptable.

Checkpoint selection and early stopping use validation `mean_daily_accuracy`, not raw validation loss. The loss still trains the network, but the saved model is the epoch with the best primary business metric.

## Policy Shift Handling

The network output is not hard-coded as `40 + softplus(z)`. Instead, it predicts real-valued prices and clips after inference using policy bounds:

```text
2025: [40, 1000]
2026: [0, 1000]
```

The config also enables conservative zero-floor augmentation. It creates low-weight synthetic samples only from already plausible low-price contexts, such as low observed prices with high renewable share, low net load, or pumped-storage charging.

## Commands

```bash
python scripts/05_train_policy_tcn.py --config config/default.yaml
python scripts/06_evaluate_policy_tcn.py --config config/default.yaml
```

Expected outputs:

```text
outputs/models/policy_aware_tcn.pt
outputs/models/policy_aware_tcn_report.json
outputs/predictions/policy_aware_tcn_predictions.csv
outputs/reports/policy_aware_tcn_daily_metrics.csv
outputs/reports/policy_aware_tcn_summary.json
outputs/predictions/policy_aware_tcn_future_24h_predictions.csv
outputs/reports/policy_aware_tcn_future_24h_summary.json
```

PyTorch is required for training and inference.
