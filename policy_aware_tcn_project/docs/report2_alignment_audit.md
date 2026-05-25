# Report2 Alignment Audit

This audit checks the standalone implementation against `../deep-research-report2.md`.

## Verdict

The standalone project implements the report's recommended first production route:

```text
TCN-based direct 96-step model
+ business-aligned loss
+ explicit policy variables
+ year-aware clipping
+ conservative 2026 zero-floor adaptation
```

It does not implement the report's second and third research routes, `LSTM seq2seq` and `PatchTST-lite`, because the report explicitly ranks them after the TCN route.

## Requirement Mapping

| Report2 item | Evidence in implementation | Status |
|---|---|---|
| Use a TCN-based direct 96-step model as the short-term main route | `src/epf_tcn/deep_models.py::PolicyAwareTCN`, direct `mu: [B, 96]` output | aligned |
| Use `history_days=14` for TCN/LSTM main config | `config/default.yaml`, `deep_model.history_days: 14` | aligned |
| Use historical observations plus known future exogenous variables | `src/epf_tcn/deep_data.py::infer_deep_feature_spec`, `x_hist` and `x_fut` groups | aligned |
| Include load, renewable, bidding space, interconnector, pumped storage, calendar, and market-derived features | `config/default.yaml` exogenous columns plus `features.py` market features | aligned |
| Include explicit static/policy variables | `policy_regime`, `price_lower_bound`, `price_upper_bound`, `is_zero_floor_regime` in `deep_data.py` | aligned |
| Do not hard-code 2025 lower bound inside the output layer | model outputs residual real values; clipping happens in `deep_train.py::predict_arrays` | aligned |
| Clip by year/policy: 2025 `[40,1000]`, 2026 `[0,1000]` | `config/default.yaml::deep_model.year_bounds` | aligned |
| Use weighted relative MAE aligned with DailyAccuracy | `deep_train.py::weighted_relative_mae_loss` | aligned |
| Add ramp regularization | `deep_train.py::weighted_relative_mae_loss`, `lambda_ramp` | aligned |
| Add optional heteroscedastic head/NLL | `deep_models.py::log_sigma_head`, `deep_train.py::gaussian_nll_loss` | aligned |
| Keep model compact, roughly 20万-50万 parameters | default smoke check reports 201506 trainable parameters | aligned |
| Use AdamW, dropout, weight decay, grad clip, early stopping | `deep_train.py::train_deep_model`, `config/default.yaml` | aligned |
| Use the user-facing daily accuracy as the primary selection metric | `deep_model.selection_metric: mean_daily_accuracy`; training saves the best validation `mean_daily_accuracy` epoch | aligned |
| Use conservative 2026 low-price synthetic augmentation | `deep_data.py::make_synthetic_zero_floor_windows` | aligned |
| Keep LightGBM as baseline/compare route | baseline remains in `../dayahead_epf_agent_project`; this project has no LightGBM training code | aligned by separation |
| Add LSTM seq2seq as second model | not implemented here | intentionally deferred |
| Add PatchTST-lite / Transformer route | not implemented here | intentionally deferred |
| Add conformal interval calibration | not implemented here | future research route |

## Implementation Details Beyond The Report

The implementation adds two engineering safeguards not explicitly written in the report:

- `y_mask` for partially missing target days, so incomplete labels do not destroy whole-day windows.
- missing-indicator channels appended to normalized historical and future inputs.

These additions are consistent with the report's warning about small sample size and data quality. They preserve more 2025 supervised days without introducing future leakage.

## Current Verified Shapes

Using `config/default.yaml` and the bundled feature table:

```text
hist_cols: 34
fut_cols: 33
x_hist after missing indicators: [B, 1344, 68]
x_fut after missing indicators: [B, 96, 66]
x_static: [B, 4]
mu/log_sigma output: [B, 96]
default trainable parameters: 201506
```

Window counts:

```text
real_train_windows: 273
valid_windows: 61
test_windows: 31
synthetic_zero_floor_windows: 263
total_train_windows: 536
```

## Conclusion

The architecture is consistent with report2's recommended TCN production route. The main differences are deliberate engineering additions for missing-label robustness and project separation from the LightGBM baseline. The LSTM, PatchTST-lite, and conformal-calibration routes remain future extensions, not omissions from the requested first architecture.
