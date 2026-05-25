# Accuracy Experiment Log

This log records no-leakage experiments aimed at the project accuracy metric:

```text
accuracy = 1 - mean(abs(y_pred - y_true) / max(abs(y_true), 40))
```

The fixed test period is 2025-12-01 through 2025-12-31 unless noted otherwise.

## Current Best Reproducible Test Result

- Config: `config/residual_calibrator_weather_error_q2.yaml`
- Command:
  ```bash
  ../dayahead_epf_agent_project/.venv/bin/python scripts/09_train_evaluate_residual_calibrator.py \
    --config config/residual_calibrator_weather_error_q2.yaml
  ```
- Output: `outputs/residual_calibrated_train2024q2/reports/state_gbm_residual_summary.json`
- Test accuracy: `0.7711041307390922`
- MAE: `90.0057497319963`
- RMSE: `177.02557964613433`
- Pass days at 85% daily accuracy: `6 / 31`

## 2025-Priority Experiment

- Config: `config/residual_calibrator_2025_priority.yaml`
- Train: 2025-01-01 through 2025-10-31
- Validation: 2025-11-01 through 2025-11-30
- Test: 2025-12-01 through 2025-12-31
- StateGBM test accuracy: `0.7605732816532624`
- Residual-calibrated test accuracy: `0.7573985596079574`

This confirms that prioritizing 2025 data improves recency alignment on 2025-11,
but it does not by itself solve the 2025-12 high-price regime shift.

## 2025 Rolling-Week Diagnostic

- Train/validation/test were rolled by week inside 2025.
- Each week used only data available before the predicted week.
- Combined 2025-12 accuracy: about `0.7053`.

The rolling approach overreacted to recent periods and raised too many normal or
low-price slots. It is not adopted as the default path.

## Negative / Ablation Results

- Scarcity rolling-state features:
  - Config: `config/residual_calibrator_scarcity_q2.yaml`
  - Residual-calibrated test accuracy: `0.7472349106265357`
  - Result: lower than the current best.
- November-only validation after training through 2025-10:
  - Config: `config/residual_calibrator_weather_error_novvalid.yaml`
  - Residual-calibrated test accuracy: `0.6770950754961472`
  - Result: validation window is too small/unstable for December.
- Simple no-leakage fusion between old-window and 2025-priority StateGBM:
  - Validation-selected fusion test accuracy: about `0.7456`
  - Test-oracle fusion capacity: about `0.7643`
  - Result: not enough complementarity to reach the current best.
- ExtraTrees diagnostic:
  - Best tested state-overridden accuracy was below `0.75`.
  - Result: still underpredicts high/cap price periods.

## Main Bottleneck

The December 2025 high-price and cap-price periods remain the bottleneck. Current
features can classify floor-price periods well, but high and cap probabilities
are not strong enough to lift predictions without damaging normal periods. The
best next improvement likely requires new leakage-safe external signals that
describe realized or forecast scarcity more directly, such as actual load,
renewable output, available capacity, reserve margin, outage/maintenance, or
official market disclosure data.
